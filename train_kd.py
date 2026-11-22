import os
import json
import copy
import math
import random
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from tqdm import tqdm

from models.efficientvit_b0 import EfficientViT_B0_SOTA
from models.modelfast_scnn import FastSCNN
from models.modelteacher_DMiTB import DeepLabV3Plus_MiTB3_LWSS_Joint
from losses.sota_kd_losses import SOTAKDLoss, KD_WEIGHTS
from eval_checkpoints import FastSCNN_KDWrapper, CWFIDEvalDataset, evaluate_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RichCWFIDTrainDataset(Dataset):
    """Unified 512x512 training dataset with scale/crop, flip, and photometric jitter."""
    def __init__(self, root, indices, target_size=(512, 512)):
        self.items = []
        for idx in indices:
            name = f"{idx:03d}_image.png"
            img = cv2.cvtColor(cv2.imread(os.path.join(root, "images", name)), cv2.COLOR_BGR2RGB)
            m = cv2.cvtColor(cv2.imread(os.path.join(root, "annotations", name.replace("_image.png", "_annotation.png"))), cv2.COLOR_BGR2RGB)
            mm = np.zeros(m.shape[:2], dtype=np.uint8)
            mm[m[:, :, 1] > 128] = 1
            mm[m[:, :, 0] > 128] = 2
            self.items.append((img, mm))
        self.ts = target_size
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, msk = self.items[i]
        h, w = img.shape[:2]
        scale = random.uniform(0.5, 1.5)
        nh, nw = int(h * scale), int(w * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (nw, nh), interpolation=cv2.INTER_NEAREST)
        th, tw = self.ts
        if nh < th or nw < tw:
            ph, pw = max(0, th - nh), max(0, tw - nw)
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            msk = cv2.copyMakeBorder(msk, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
            nh, nw = img.shape[:2]
        y0 = random.randint(0, nh - th)
        x0 = random.randint(0, nw - tw)
        img = img[y0:y0 + th, x0:x0 + tw]
        msk = msk[y0:y0 + th, x0:x0 + tw]
        if random.random() < 0.5:
            img, msk = np.fliplr(img), np.fliplr(msk)
        if random.random() < 0.5:
            img, msk = np.flipud(img), np.flipud(msk)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        if random.random() < 0.5:
            x = (x * random.uniform(0.85, 1.15)).clamp(0, 1)
        return self.normalize(x), torch.from_numpy(np.ascontiguousarray(msk).astype(np.int64))


def cutmix_batch(x, y, p=0.5):
    if p <= 0 or random.random() > p:
        return x, y
    B, _, H, W = x.shape
    perm = torch.randperm(B, device=x.device)
    frac = random.uniform(0.10, 0.50)
    ch, cw = int(H * math.sqrt(frac)), int(W * math.sqrt(frac))
    y0, x0 = random.randint(0, H - ch), random.randint(0, W - cw)
    x, y = x.clone(), y.clone()
    x[:, :, y0:y0 + ch, x0:x0 + cw] = x[perm][:, :, y0:y0 + ch, x0:x0 + cw]
    y[:, y0:y0 + ch, x0:x0 + cw] = y[perm][:, y0:y0 + ch, x0:x0 + cw]
    return x, y


class EMA:
    def __init__(self, model, max_decay=0.995):
        self.ema = copy.deepcopy(model).eval()
        self.max_decay, self.n = max_decay, 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.max_decay, (1.0 + self.n) / (10.0 + self.n))
        for ep, mp in zip(self.ema.parameters(), model.parameters()):
            ep.mul_(d).add_(mp.detach(), alpha=1.0 - d)
        for eb, mb in zip(self.ema.buffers(), model.buffers()):
            eb.copy_(mb)


class DiceLoss(nn.Module):
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        oh = F.one_hot(targets.clamp(0, 2), 3).permute(0, 3, 1, 2).float()
        inter = (probs * oh).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + oh.sum(dim=(0, 2, 3))
        return (1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def main():
    parser = argparse.ArgumentParser(description="Reconstruct CWFID 6-Method SOTA KD Benchmark")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--student", type=str, choices=["EfficientViT_B0", "Fast_SCNN"], required=True)
    parser.add_argument("--kd_method", type=str, choices=list(KD_WEIGHTS.keys()), required=True)
    parser.add_argument("--split_json", type=str, default="splits/split_stratified.json")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="runs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    with open(args.split_json, "r") as f:
        sp = json.load(f)["split"]

    teacher = DeepLabV3Plus_MiTB3_LWSS_Joint(3).to(device).eval()
    sd = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    teacher.load_state_dict(sd, strict=False)
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = (EfficientViT_B0_SOTA(3) if args.student == "EfficientViT_B0" else FastSCNN_KDWrapper(3)).to(device)
    ema = EMA(student, max_decay=0.995)
    kd_cfg = KD_WEIGHTS[args.kd_method]
    kd_fn = SOTAKDLoss(args.kd_method, feat_channels=128, temperature=kd_cfg["T"]).to(device)

    params = list(student.parameters()) + (list(kd_fn.parameters()) if args.kd_method == "MGD" else [])
    optimizer = optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    dice_fn = DiceLoss()
    focal_fn = FocalLoss(gamma=2.0, weight=torch.tensor([0.05, 0.45, 0.50], device=device))

    train_ds = RichCWFIDTrainDataset(args.data_root, sp["train"])
    g = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler([1.0 / len(train_ds)] * len(train_ds), num_samples=args.steps * 8, replacement=True, generator=g)
    loader = DataLoader(train_ds, batch_size=8, sampler=sampler, drop_last=True, num_workers=2)
    test_loader = DataLoader(CWFIDEvalDataset(args.data_root, sp["test"]), batch_size=4, shuffle=False)

    for step, (images, masks) in tqdm(enumerate(loader), total=args.steps):
        student.train()
        lf = (0.1 + 0.9 * (step + 1) / 300) if step < 300 else (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - 300) / max(1, args.steps - 300))))
        for pg in optimizer.param_groups:
            pg["lr"] = 1e-3 * lf

        images, masks = cutmix_batch(images.to(device), masks.to(device), p=0.5)
        s_logits, _, s_fused = student(images, return_kd=True)
        with torch.no_grad():
            t_logits, t_feat = teacher(images, return_kd_features=True)

        l_hard = 0.45 * dice_fn(s_logits, masks) + 0.45 * focal_fn(s_logits, masks)
        l_kd = kd_fn(s_logits, t_logits, s_fused, t_feat, masks)
        loss = l_hard + kd_cfg["w_kd"] * l_kd

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        ema.update(student)

    res = evaluate_model(ema.ema, test_loader, device)
    out_ckpt = os.path.join(args.out_dir, f"{args.student}_{args.kd_method}_s{args.seed}_last_ema.pth")
    torch.save(ema.ema.state_dict(), out_ckpt)
    print(f"Saved: {out_ckpt} | Test Results: {res}")


if __name__ == "__main__":
    main()

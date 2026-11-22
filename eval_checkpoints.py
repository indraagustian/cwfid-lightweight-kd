import os
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from models.efficientvit_b0 import EfficientViT_B0_SOTA
from models.modelfast_scnn import FastSCNN


class FastSCNN_KDWrapper(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.base = FastSCNN(num_classes=num_classes)

    def forward(self, x, return_kd=False):
        size = x.size()[2:]
        higher_res = self.base.learning_to_downsample(x)
        low_res = self.base.global_feature_extractor(higher_res)
        fused = self.base.feature_fusion(higher_res, low_res)
        logits = F.interpolate(self.base.classifier(fused), size=size, mode="bilinear", align_corners=True)
        if return_kd:
            feat_up = F.interpolate(fused, size=size, mode="bilinear", align_corners=False)
            return logits, feat_up, fused
        return logits


class CWFIDEvalDataset(Dataset):
    def __init__(self, root, indices, target_size=(512, 512)):
        self.root = root
        self.indices = indices
        self.ts = target_size
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        name = f"{idx:03d}_image.png"
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root, "images", name)), cv2.COLOR_BGR2RGB)
        m = cv2.cvtColor(cv2.imread(os.path.join(self.root, "annotations", name.replace("_image.png", "_annotation.png"))), cv2.COLOR_BGR2RGB)
        mm = np.zeros(m.shape[:2], dtype=np.uint8)
        mm[m[:, :, 1] > 128] = 1  # Crop
        mm[m[:, :, 0] > 128] = 2  # Weed
        th, tw = self.ts
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        mm = cv2.resize(mm, (tw, th), interpolation=cv2.INTER_NEAREST)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return self.normalize(x), torch.from_numpy(mm.astype(np.int64))


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    cm = torch.zeros(3, 3, device=device, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        cm += torch.bincount((y * 3 + p).flatten(), minlength=9).view(3, 3)
    cm = cm.cpu().numpy().astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    iou = tp / (tp + fp + fn + 1e-12) * 100.0
    prec = tp / (tp + fp + 1e-12) * 100.0
    rec = tp / (tp + fn + 1e-12) * 100.0
    return dict(
        mIoU=round(float(iou.mean()), 2),
        Soil_IoU=round(float(iou[0]), 2),
        Crop_IoU=round(float(iou[1]), 2),
        Weed_IoU=round(float(iou[2]), 2),
        mIoU_cw=round(float(0.5 * (iou[1] + iou[2])), 2),
        Weed_Prec=round(float(prec[2]), 2),
        Weed_Rec=round(float(rec[2]), 2),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate 12 pre-trained KD checkpoints on CWFID SPLIT_OPT test set")
    parser.add_argument("--data_root", type=str, required=True, help="Path to CWFID dataset-1.0 directory")
    parser.add_argument("--split_json", type=str, default="splits/split_stratified.json")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.split_json, "r") as f:
        test_idx = json.load(f)["split"]["test"]

    loader = DataLoader(CWFIDEvalDataset(args.data_root, test_idx), batch_size=4, shuffle=False)
    methods = ["Vanilla_KL", "IFVD", "DIST", "MGD", "CWD", "AT"]
    rows = []

    for stu in ["EfficientViT_B0", "Fast_SCNN"]:
        for m in methods:
            ckpt_path = os.path.join(args.ckpt_dir, f"{stu}_{m}_s42_last_ema.pth")
            if not os.path.exists(ckpt_path):
                print(f"[SKIP] Missing: {ckpt_path}")
                continue
            model = EfficientViT_B0_SOTA(3) if stu == "EfficientViT_B0" else FastSCNN_KDWrapper(3)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.to(device)
            res = evaluate_model(model, loader, device)
            rows.append({"Student": stu, "Method": m, **res})

    if rows:
        cols = list(rows[0].keys())
        widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
        header = "  ".join(f"{c:<{widths[c]}}" for c in cols)
        print(header)
        print("-" * len(header))
        for r in rows:
            print("  ".join(f"{str(r[c]):<{widths[c]}}" for c in cols))


if __name__ == "__main__":
    main()

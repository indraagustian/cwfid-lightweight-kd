import torch
import torch.nn as nn
import torch.nn.functional as F


class SOTAKDLoss(nn.Module):
    """
    Six Knowledge Distillation Loss Formulations for Semantic Segmentation:
      1. Vanilla_KL : Hinton et al. (2015) - pixel-wise KL(T=4.0) * T^2
      2. AT         : Zagoruyko & Komodakis (ICLR 2017) - L2-normalized spatial attention transfer
      3. CWD        : Shu et al. (ICCV 2021) - Channel-Wise spatial softmax KL(T=4.0)
      4. MGD        : Yang et al. (ECCV 2022) - Masked Generative feature reconstruction
      5. DIST       : Huang et al. (NeurIPS 2022) - Inter + Intra-class Pearson correlation
      6. IFVD       : Wang et al. (ECCV 2020) - Intra-Class Feature Variation prototype similarity
    """
    def __init__(self, method, feat_channels=128, temperature=4.0, mgd_mask_ratio=0.65):
        super().__init__()
        self.method = method
        self.T = temperature
        self.mgd_mask_ratio = mgd_mask_ratio
        if method == "MGD":
            self.mgd_gen = nn.Sequential(
                nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            )

    @staticmethod
    def _pearson_dist(u, v, dim=-1, eps=1e-6):
        u_c = u - u.mean(dim=dim, keepdim=True)
        v_c = v - v.mean(dim=dim, keepdim=True)
        num = (u_c * v_c).sum(dim=dim)
        den = torch.sqrt((u_c.pow(2).sum(dim=dim) + eps) * (v_c.pow(2).sum(dim=dim) + eps))
        return (1.0 - num / den).mean()

    def forward(self, s_logits, t_logits, s_feat, t_feat, masks):
        if self.method == "Vanilla_KL":
            p_s = F.log_softmax(s_logits / self.T, dim=1)
            p_t = F.softmax(t_logits / self.T, dim=1)
            kl = F.kl_div(p_s, p_t, reduction="none").sum(dim=1).mean()
            return kl * (self.T ** 2)

        elif self.method == "AT":
            a_s = s_feat.pow(2).mean(dim=1)
            a_t = t_feat.pow(2).mean(dim=1)
            a_s = F.normalize(a_s.flatten(1), p=2, dim=1)
            a_t = F.normalize(a_t.flatten(1), p=2, dim=1)
            return (a_s - a_t).pow(2).sum(dim=1).mean()

        elif self.method == "CWD":
            B, C, H, W = s_feat.shape
            s_flat = s_feat.view(B, C, -1) / self.T
            t_flat = t_feat.view(B, C, -1) / self.T
            p_s = F.log_softmax(s_flat, dim=-1)
            p_t = F.softmax(t_flat, dim=-1)
            cwd_loss = F.kl_div(p_s, p_t, reduction="none").sum(dim=-1).mean()
            return cwd_loss * (self.T ** 2)

        elif self.method == "MGD":
            B, C, H, W = s_feat.shape
            mat = torch.rand((B, 1, H, W), device=s_feat.device)
            mask = (mat > self.mgd_mask_ratio).float()
            s_gen = self.mgd_gen(s_feat * mask)
            return F.mse_loss(s_gen, t_feat)

        elif self.method == "DIST":
            p_s = F.softmax(s_logits / self.T, dim=1)
            p_t = F.softmax(t_logits / self.T, dim=1)
            inter = self._pearson_dist(p_s.permute(0, 2, 3, 1), p_t.permute(0, 2, 3, 1), dim=-1)
            intra = self._pearson_dist(p_s.flatten(2), p_t.flatten(2), dim=-1)
            return inter + intra

        elif self.method == "IFVD":
            m_small = F.interpolate(masks.unsqueeze(1).float(), size=s_feat.shape[2:], mode="nearest").squeeze(1).long()
            s_n = F.normalize(s_feat.flatten(2), p=2, dim=1)
            t_n = F.normalize(t_feat.flatten(2), p=2, dim=1)
            m_flat = m_small.flatten(1)
            loss_total = s_feat.new_tensor(0.0)
            valid_cnt = 0
            for cls_id in range(3):
                cls_mask = (m_flat == cls_id).unsqueeze(1).float()
                denom = cls_mask.sum(dim=-1, keepdim=True)
                if (denom > 0).any():
                    proto_s = F.normalize((s_n * cls_mask).sum(dim=-1, keepdim=True) / (denom + 1e-6), p=2, dim=1)
                    proto_t = F.normalize((t_n * cls_mask).sum(dim=-1, keepdim=True) / (denom + 1e-6), p=2, dim=1)
                    sim_s = (s_n * proto_s).sum(dim=1)
                    sim_t = (t_n * proto_t).sum(dim=1)
                    loss_total = loss_total + F.mse_loss(sim_s, sim_t)
                    valid_cnt += 1
            return loss_total / max(valid_cnt, 1)

        raise ValueError(f"Unknown KD method: {self.method}")


KD_WEIGHTS = {
    "Vanilla_KL": dict(w_kd=0.50, T=4.0),
    "AT":         dict(w_kd=100.0, T=4.0),
    "CWD":        dict(w_kd=1.00, T=4.0),
    "MGD":        dict(w_kd=0.50, T=4.0),
    "DIST":       dict(w_kd=1.00, T=4.0),
    "IFVD":       dict(w_kd=10.0, T=4.0),
}

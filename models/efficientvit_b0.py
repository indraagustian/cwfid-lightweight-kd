import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=None, g=1, act=True):
        if p is None:
            p = k // 2
        layers = [nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False), nn.BatchNorm2d(out_c)]
        if act:
            layers.append(nn.Hardswish(inplace=True))
        super().__init__(*layers)


class MBConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expand=4):
        super().__init__()
        mid = in_c * expand
        self.use_res = (stride == 1 and in_c == out_c)
        self.block = nn.Sequential(
            ConvBNAct(in_c, mid, k=1, s=1, p=0, act=True),
            ConvBNAct(mid, mid, k=3, s=stride, p=1, g=mid, act=True),
            ConvBNAct(mid, out_c, k=1, s=1, p=0, act=False),
        )

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


class LiteMSA(nn.Module):
    """Lightweight Multi-Scale Linear Attention (Cai et al., ICCV 2023)."""
    def __init__(self, dim, head_dim=16, scales=(5,)):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = max(1, dim // head_dim)
        total_dim = self.num_heads * head_dim
        self.qkv = nn.Conv2d(dim, 3 * total_dim, 1, bias=False)
        self.aggreg = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3 * total_dim, 3 * total_dim, s, padding=s // 2, groups=3 * total_dim, bias=False),
                nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * self.num_heads, bias=False),
            ) for s in scales
        ])
        self.proj = ConvBNAct(total_dim * (1 + len(scales)), dim, k=1, s=1, p=0, act=False)
        self.eps = 1e-6

    def _relu_linear_attn(self, qkv):
        B, C3, H, W = qkv.shape
        N = H * W
        qkv = qkv.reshape(B, -1, 3 * self.head_dim, N)
        q, k, v = qkv.chunk(3, dim=2)
        q = F.relu(q)
        k = F.relu(k)
        v_pad = F.pad(v, (0, 0, 0, 1), value=1.0)
        kv = torch.matmul(v_pad, k.transpose(-1, -2))
        out = torch.matmul(kv, q)
        out = out[:, :, :-1, :] / (out[:, :, -1:, :] + self.eps)
        return out.reshape(B, -1, H, W)

    def forward(self, x):
        qkv0 = self.qkv(x)
        multi = [qkv0] + [op(qkv0) for op in self.aggreg]
        out = torch.cat([self._relu_linear_attn(m) for m in multi], dim=1)
        return x + self.proj(out)


class EfficientViTBlock(nn.Module):
    def __init__(self, dim, head_dim=16, scales=(5,), expand=4):
        super().__init__()
        self.attn = LiteMSA(dim, head_dim=head_dim, scales=scales)
        self.mbconv = MBConv(dim, dim, stride=1, expand=expand)

    def forward(self, x):
        return self.mbconv(self.attn(x))


class EfficientViT_B0_SOTA(nn.Module):
    """EfficientViT-B0 configured for 3-class crop-weed segmentation (0.405M params)."""
    def __init__(self, num_classes=3, kd_channels=128):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 8, k=3, s=2),
            MBConv(8, 8, stride=1, expand=4),
        )
        self.stage1 = nn.Sequential(MBConv(8, 16, stride=2), MBConv(16, 16, stride=1))
        self.stage2 = nn.Sequential(MBConv(16, 32, stride=2), MBConv(32, 32, stride=1))
        self.stage3 = nn.Sequential(MBConv(32, 64, stride=2), EfficientViTBlock(64, head_dim=16))
        self.stage4 = nn.Sequential(MBConv(64, 128, stride=2), EfficientViTBlock(128, head_dim=16))

        self.p2_proj = ConvBNAct(32, 32, k=1, s=1, p=0, act=False)
        self.p3_proj = ConvBNAct(64, 32, k=1, s=1, p=0, act=False)
        self.p4_proj = ConvBNAct(128, 32, k=1, s=1, p=0, act=False)
        self.head_fuse = nn.Sequential(
            MBConv(32, 32, stride=1, expand=4),
            ConvBNAct(32, 32, k=1, s=1, p=0, act=True),
        )
        self.classifier = nn.Conv2d(32, num_classes, 1)

        # Training-only projection (32 -> 128 channels) for KD & PCL; discarded at inference
        self.pcl_proj = nn.Sequential(
            nn.Conv2d(32, kd_channels, 1, bias=False),
            nn.BatchNorm2d(kd_channels),
        )

        # Optimizer parameter groups (kept for exact checkpoint state_dict compatibility)
        self.ltd = nn.ModuleList([self.stem, self.stage1])
        self.gfe = nn.ModuleList([self.stage2, self.stage3, self.stage4])
        self.ffm = nn.Identity()
        self.decoder = nn.ModuleList([
            self.p2_proj, self.p3_proj, self.p4_proj,
            self.head_fuse, self.classifier, self.pcl_proj,
        ])

    def forward(self, x, return_kd=False):
        size = x.shape[2:]
        x = self.stem(x)
        x = self.stage1(x)
        p2 = self.stage2(x)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)

        h8_size = p2.shape[2:]
        f = (
            self.p2_proj(p2)
            + F.interpolate(self.p3_proj(p3), size=h8_size, mode="bilinear", align_corners=False)
            + F.interpolate(self.p4_proj(p4), size=h8_size, mode="bilinear", align_corners=False)
        )
        f = self.head_fuse(f)
        logits = F.interpolate(self.classifier(f), size=size, mode="bilinear", align_corners=False)

        if return_kd:
            fused = self.pcl_proj(f)
            feat_up = F.interpolate(fused, size=size, mode="bilinear", align_corners=False)
            return logits, feat_up, fused
        return logits

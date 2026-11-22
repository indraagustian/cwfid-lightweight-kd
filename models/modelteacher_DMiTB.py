import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class ExGConcatenationFusion(nn.Module):
    """Mengekstrak fitur klorofil ExG (2G - R - B) dan menggabungkannya ke low-level feature."""
    def __init__(self, rgb_channels=48, exg_channels=16):
        super().__init__()
        self.exg_Encoder = nn.Sequential(
            nn.Conv2d(1, exg_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(exg_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(exg_channels, exg_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(exg_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(rgb_channels + exg_channels, rgb_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True)
        )
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_exg(self, x_norm):
        x_rgb = torch.clamp(x_norm * self.std + self.mean, 0.0, 1.0)
        r, g, b = x_rgb[:, 0:1], x_rgb[:, 1:2], x_rgb[:, 2:3]
        exg = 2.0 * g - r - b
        return exg

    def forward(self, low_level_rgb, x_input):
        exg = self.compute_exg(x_input)
        exg_feat = self.exg_Encoder(exg)
        if exg_feat.shape[2:] != low_level_rgb.shape[2:]:
            exg_feat = F.interpolate(exg_feat, size=low_level_rgb.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([low_level_rgb, exg_feat], dim=1)
        return self.fusion_conv(fused)

class DeepLabV3Plus_MiTB3_LWSS_Joint(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.base_model = smp.DeepLabV3Plus(
            encoder_name="mit_b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36)
        )
        self.exg_fusion = ExGConcatenationFusion(rgb_channels=48, exg_channels=16)
        self.edge_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, bias=True)
        )

    def forward(self, x, return_edge=False, return_kd_features=False):
        features = self.base_model.encoder(x)
        aspp_out = self.base_model.decoder.aspp(features[-1])
        aspp_up = F.interpolate(aspp_out, size=features[2].shape[2:], mode='bilinear', align_corners=False)

        low_level_proj = self.base_model.decoder.block1(features[2]) 
        low_level_fused = self.exg_fusion(low_level_proj, x)         

        concat_features = torch.cat([aspp_up, low_level_fused], dim=1)
        decoder_out = self.base_model.decoder.block2(concat_features) 

        logits = self.base_model.segmentation_head(decoder_out)       

        if return_edge and return_kd_features:
            edge_logits = F.interpolate(self.edge_head(decoder_out), size=x.shape[2:], mode='bilinear', align_corners=False)
            return logits, edge_logits, features[3]
        elif return_edge:
            edge_logits = F.interpolate(self.edge_head(decoder_out), size=x.shape[2:], mode='bilinear', align_corners=False)
            return logits, edge_logits
        elif return_kd_features:
            return logits, features[3] 
        return logits
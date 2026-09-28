"""DeepLabV3+ segmentation decoder (ASPP + low-level skip connection).

Plain image-in / per-pixel-class-logits-out decoder, no text conditioning: this stage
only tests whether the frozen Qwen3-VL vision tower's features are useful for Mars
terrain segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.pool(x)
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, atrous_rates=(6, 12, 18)):
        super().__init__()
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        r1, r2, r3 = atrous_rates
        self.branch_r1 = ASPPConv(in_channels, out_channels, r1)
        self.branch_r2 = ASPPConv(in_channels, out_channels, r2)
        self.branch_r3 = ASPPConv(in_channels, out_channels, r3)
        self.branch_pool = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [
            self.branch_1x1(x),
            self.branch_r1(x),
            self.branch_r2(x),
            self.branch_r3(x),
            self.branch_pool(x),
        ]
        return self.project(torch.cat(feats, dim=1))


class DeepLabV3PlusDecoder(nn.Module):
    """high_level -> ASPP -> upsample 4x -> concat with projected low_level -> classifier."""

    def __init__(
        self,
        num_classes: int,
        high_level_channels: int,
        low_level_channels: int,
        aspp_out_channels: int = 256,
        low_level_proj_channels: int = 48,
    ):
        super().__init__()
        self.aspp = ASPP(high_level_channels, aspp_out_channels)

        self.low_level_project = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_proj_channels, 1, bias=False),
            nn.BatchNorm2d(low_level_proj_channels),
            nn.ReLU(inplace=True),
        )

        fused_channels = aspp_out_channels + low_level_proj_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels, aspp_out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_out_channels, aspp_out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_out_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(aspp_out_channels, num_classes, 1)

    def forward(self, low_level: torch.Tensor, high_level: torch.Tensor, output_size) -> torch.Tensor:
        x = self.aspp(high_level)
        x = F.interpolate(x, size=low_level.shape[-2:], mode="bilinear", align_corners=False)

        low = self.low_level_project(low_level)
        x = torch.cat([x, low], dim=1)
        x = self.fuse(x)
        x = self.classifier(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

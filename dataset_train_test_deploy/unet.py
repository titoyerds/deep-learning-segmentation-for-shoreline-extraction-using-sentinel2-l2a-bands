import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetEncoder(nn.Module):
    """
    ResNet-34 encoder with pretrained ImageNet weights.
    Since Sentinel-2 has 12 bands (not 3), we replace the first conv layer.
    """
    def __init__(self, num_bands):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
 
        # Replace first conv: 3-channel → num_bands, stride=1 keeps full input resolution.
        # For 512×512 input: e0 stays 512×512 (full-res skip for the last decoder step).
        self.encoder0 = nn.Sequential(
            nn.Conv2d(num_bands, 64, kernel_size=7, stride=1, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
        )  # output: 64 ch, H×W (same as input)

        self.pool     = resnet.maxpool  # H/2 × W/2  (e.g. 512→256)

        self.encoder1 = resnet.layer1   # 64  ch, H/2  × W/2   (e.g. 256×256)
        self.encoder2 = resnet.layer2   # 128 ch, H/4  × W/4   (e.g. 128×128)
        self.encoder3 = resnet.layer3   # 256 ch, H/8  × W/8   (e.g.  64×64)
        self.encoder4 = resnet.layer4   # 512 ch, H/16 × W/16  (e.g.  32×32)

    def forward(self, x):
        e0 = self.encoder0(x)       # 64ch,  H    × W     (e.g. 512×512) ← full-res skip
        p  = self.pool(e0)          # 64ch,  H/2  × W/2   (e.g. 256×256)
        e1 = self.encoder1(p)       # 64ch,  H/2  × W/2   (e.g. 256×256)
        e2 = self.encoder2(e1)      # 128ch, H/4  × W/4   (e.g. 128×128)
        e3 = self.encoder3(e2)      # 256ch, H/8  × W/8   (e.g.  64×64)
        e4 = self.encoder4(e3)      # 512ch, H/16 × W/16  (e.g.  32×32)
        return e0, e1, e2, e3, e4
    

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# UNet with ResNet-34 Encoder
 
class UNet(nn.Module):
    """
    Decoder path mirrors the encoder resolutions (example for 512x512 input):
      e4: 512 ch, H/16 x W/16  (e.g.  32x32)
      e3: 256 ch, H/8  x W/8   (e.g.  64x64)
      e2: 128 ch, H/4  x W/4   (e.g. 128x128)
      e1:  64 ch, H/2  x W/2   (e.g. 256x256)
      e0:  64 ch, H    x W     (e.g. 512x512) ← full-resolution skip

    Each up-block doubles spatial resolution via bilinear upsampling + skip concat.
    Output resolution matches input resolution for any HxW input.
    """
    def __init__(self, num_classes, num_bands):
        super().__init__()

        self.encoder = ResNetEncoder(num_bands)

        # Decoder: each block receives upsampled features + matching skip
        self.up4 = DoubleConv(512 + 256, 256)   # H/16 → H/8,  cat e3
        self.up3 = DoubleConv(256 + 128, 128)   # H/8  → H/4,  cat e2
        self.up2 = DoubleConv(128 + 64,   64)   # H/4  → H/2,  cat e1
        self.up1 = DoubleConv( 64 + 64,   64)   # H/2  → H,    cat e0

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e0, e1, e2, e3, e4 = self.encoder(x)

        u4 = F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False)
        u4 = self.up4(torch.cat([u4, e3], dim=1))  # 256ch, H/8  × W/8

        u3 = F.interpolate(u4, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.up3(torch.cat([u3, e2], dim=1))  # 128ch, H/4  × W/4

        u2 = F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, e1], dim=1))  # 64ch,  H/2  × W/2

        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, e0], dim=1))  # 64ch,  H    × W

        return self.final(u1)


def dice_loss(pred, target, smooth=1.0):
    pred = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2)
 
    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union        = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
 
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


class DiceCELoss(nn.Module):
    """
    Dice + CrossEntropy hybrid loss.
    total = alpha * DiceLoss + ce_weight * CrossEntropyLoss

    Dice handles region-level overlap; CE provides dense per-pixel supervision
    with shoreline class up-weighting to counter class imbalance.
    Sobel/boundary term removed — Dice already captures boundary quality
    implicitly, and the hybrid is simpler to tune and reproduce.
    """
    def __init__(self, alpha=1.0, ce_weight=0.5):
        super().__init__()
        self.dice      = dice_loss
        self.alpha     = alpha
        self.ce_weight = ce_weight
        # Weight class 1 (shoreline) 9x more than class 0 (background)
        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor([0.1, 0.9])
        )

    def forward(self, pred, target, return_components=False):
        d_loss = self.dice(pred, target)
        c_loss = self.ce(pred, target.long())
        total  = self.alpha * d_loss + self.ce_weight * c_loss

        if return_components:
            return total, d_loss, c_loss
        return total
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from blocks import ResidualBlock
from ptflops import get_model_complexity_info


class NanoNet(nn.Module):
    def __init__(self):
        super(NanoNet, self).__init__()

        self.backbone = torchvision.models.mobilenet_v2(
            # weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2,
        )
        del self.backbone.classifier, self.backbone.features[4:]

    def forward(self, x):
        x0 = self.backbone.features[0:2](x)
        x1 = self.backbone.features[2:](x0)
        x2 = self.residual_block0(x1)

        x3 = F.interpolate(x2, scale_factor=2, mode="bilinear")
        x1 = self.skip0(x1)
        diffY = x1.size()[2] - x3.size()[2]
        diffX = x1.size()[3] - x3.size()[3]
        x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x3 = torch.cat([x3, x1], 1)
        x3 = self.residual_block1(x3)

        x4 = F.interpolate(x3, scale_factor=2, mode="bilinear")
        x0 = self.skip1(x0)
        diffY = x0.size()[2] - x4.size()[2]
        diffX = x0.size()[3] - x4.size()[3]
        x4 = F.pad(x4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x4 = torch.cat([x4, x0], 1)
        x4 = self.residual_block2(x4)

        x5 = F.interpolate(x4, scale_factor=2, mode="bilinear")
        x = self.skip2(x)
        diffY = x.size()[2] - x5.size()[2]
        diffX = x.size()[3] - x5.size()[3]
        x5 = F.pad(x5, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x5 = torch.cat([x5, x], 1)
        x5 = self.residual_block3(x5)

        out = self.out(x5)

        return out


class NanoNetA(NanoNet):
    def __init__(self):
        super(NanoNetA, self).__init__()

        self.residual_block0 = ResidualBlock(24, 192)
        self.residual_block1 = ResidualBlock(320, 128)
        self.residual_block2 = ResidualBlock(192, 64)
        self.residual_block3 = ResidualBlock(96, 32)

        self.skip0 = nn.Sequential(
            nn.Conv2d(24, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(16, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(3, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Conv2d(32, 1, 1)


class NanoNetB(NanoNet):
    def __init__(self):
        super(NanoNetB, self).__init__()

        self.residual_block0 = ResidualBlock(24, 128)
        self.residual_block1 = ResidualBlock(224, 96)
        self.residual_block2 = ResidualBlock(160, 64)
        self.residual_block3 = ResidualBlock(96, 32)

        self.skip0 = nn.Sequential(
            nn.Conv2d(24, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(16, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(3, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Conv2d(32, 1, 1)


class NanoNetC(NanoNet):
    def __init__(self):
        super(NanoNetC, self).__init__()

        self.residual_block0 = ResidualBlock(24, 48)
        self.residual_block1 = ResidualBlock(80, 32)
        self.residual_block2 = ResidualBlock(56, 24)
        self.residual_block3 = ResidualBlock(40, 16)

        self.skip0 = nn.Sequential(
            nn.Conv2d(24, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(16, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Conv2d(16, 1, 1)


def calculate_param_flops(model):
    macs, params = get_model_complexity_info(
        model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))


if __name__ == "__main__":
    for model in [NanoNetA, NanoNetB, NanoNetC]:
        print(model().__str__())
        calculate_param_flops(model())
        print("+++++++++++++++++++++++++++")

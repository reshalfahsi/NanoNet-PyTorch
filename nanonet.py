import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from blocks import ResidualBlock, ConvNormAct
from ptflops import get_model_complexity_info


class NanoNet(nn.Module):
    def __init__(self):
        super(NanoNet, self).__init__()

        self.conv1 = nn.Sequential(
            ConvNormAct(3, 16, 3, stride=2),
            ConvNormAct(16, 16, 3, groups=16),
            ConvNormAct(16, 8, 1, act=False),
            ConvNormAct(8, 48, 1),
        )
        self.conv2 = nn.Sequential(
            ConvNormAct(48, 48, 3, stride=2, groups=48),
            ConvNormAct(48, 8, 1, act=False),
        )
        self.conv3 = nn.Sequential(
            ConvNormAct(8, 48, 1),
            ConvNormAct(48, 48, 3, groups=48),
            ConvNormAct(48, 8, 1, act=False),
        )
        self.conv4 = ConvNormAct(8, 48, 1)
        self.conv5 = nn.Sequential(
            ConvNormAct(48, 48, 3, stride=2, groups=48),
            ConvNormAct(48, 16, 1, act=False),
        )
        self.conv6 = nn.Sequential(
            ConvNormAct(16, 96, 1),
            ConvNormAct(96, 96, 3, groups=96),
            ConvNormAct(96, 16, 1, act=False),
        )
        self.conv7 = nn.Sequential(
            ConvNormAct(16, 96, 1),
            ConvNormAct(96, 96, 3, groups=96),
            ConvNormAct(96, 16, 1, act=False),
        )
        self.conv8 = ConvNormAct(16, 96, 1)

    def forward(self, x):

        x0 = x
        x1 = self.conv1(x0)

        y = self.conv2(x1)
        y = y + self.conv3(y)
        x2 = self.conv4(y)

        y = self.conv5(x2)
        y = y + self.conv6(y)
        y = y + self.conv7(y)
        y = self.conv8(y)

        x3 = self.residual_block0(y)

        x4 = F.interpolate(x3, scale_factor=2, mode="bilinear")
        x2 = self.skip0(x2)
        diffY = x2.size()[2] - x4.size()[2]
        diffX = x2.size()[3] - x4.size()[3]
        x4 = F.pad(x4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x4 = torch.cat([x4, x2], 1)
        x4 = self.residual_block1(x4)

        x5 = F.interpolate(x4, scale_factor=2, mode="bilinear")
        x1 = self.skip1(x1)
        diffY = x1.size()[2] - x5.size()[2]
        diffX = x1.size()[3] - x5.size()[3]
        x5 = F.pad(x5, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x5 = torch.cat([x5, x1], 1)
        x5 = self.residual_block2(x5)

        x6 = F.interpolate(x5, scale_factor=2, mode="bilinear")
        x0 = self.skip2(x0)
        diffY = x0.size()[2] - x6.size()[2]
        diffX = x0.size()[3] - x6.size()[3]
        x6 = F.pad(x6, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x6 = torch.cat([x6, x0], 1)
        x6 = self.residual_block3(x6)

        out = self.out(x6)

        return out


class NanoNetA(NanoNet):
    def __init__(self):
        super(NanoNetA, self).__init__()

        self.residual_block0 = ResidualBlock(96, 192)
        self.residual_block1 = ResidualBlock(320, 128)
        self.residual_block2 = ResidualBlock(192, 64)
        self.residual_block3 = ResidualBlock(96, 32)

        self.skip0 = nn.Sequential(
            nn.Conv2d(48, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(48, 64, 1),
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

        self.residual_block0 = ResidualBlock(96, 128)
        self.residual_block1 = ResidualBlock(224, 96)
        self.residual_block2 = ResidualBlock(160, 64)
        self.residual_block3 = ResidualBlock(96, 32)

        self.skip0 = nn.Sequential(
            nn.Conv2d(48, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(48, 64, 1),
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

        self.residual_block0 = ResidualBlock(96, 48)
        self.residual_block1 = ResidualBlock(80, 32)
        self.residual_block2 = ResidualBlock(56, 24)
        self.residual_block3 = ResidualBlock(40, 16)

        self.skip0 = nn.Sequential(
            nn.Conv2d(48, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(48, 24, 1),
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Callable, List, Optional, Sequence, Tuple, Union


class ConvNormAct(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        act=True,
        norm=True,
    ):
        super(ConvNormAct, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels, 
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super(SqueezeExcitation, self).__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1, bias=False)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1, bias=False)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, expansion_channels):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, expansion_channels // 4, 1),
            nn.BatchNorm2d(expansion_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(expansion_channels // 4, expansion_channels // 4, 3, padding=1),
            nn.BatchNorm2d(expansion_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(expansion_channels // 4, expansion_channels, 3, padding=1),
            nn.BatchNorm2d(expansion_channels),
        )

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, expansion_channels, 1, padding=0),
            nn.BatchNorm2d(expansion_channels),
        )

        self.se = SqueezeExcitation(expansion_channels, expansion_channels // 16)

    def forward(self, x):
        out = self.conv(x)
        out = out + self.skip(x)
        out = torch.relu_(out)
        out = self.se(out)
        return out

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from fling.utils.registry_utils import MODEL_REGISTRY

class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt Bottleneck Block type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua).
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 cardinality,
                 base_width,
                 stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()

        dim = int(math.floor(planes * (base_width / 64.0)))

        self.conv_reduce = nn.Conv2d(
            inplanes,
            dim * cardinality,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn_reduce = nn.BatchNorm2d(dim * cardinality)

        self.conv_conv = nn.Conv2d(
            dim * cardinality,
            dim * cardinality,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn = nn.BatchNorm2d(dim * cardinality)

        self.conv_expand = nn.Conv2d(
            dim * cardinality,
            planes * 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn_expand = nn.BatchNorm2d(planes * 4)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):
    """ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf."""

    def __init__(self, block, depth, cardinality, base_width, class_number):
        super(CifarResNeXt, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        layer_blocks = (depth - 2) // 9

        self.cardinality = cardinality
        self.base_width = base_width
        self.num_classes = class_number

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)

        self.inplanes = 64
        self.stage_1 = self._make_layer(block, 64, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(256 * block.expansion, class_number)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, self.cardinality, self.base_width, stride,
                  downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, self.cardinality, self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x, mode: str = 'compute-logit'):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        if mode == 'compute-logit':
            return y
        elif mode == 'compute-feature-logit':
            return x, y
        else:
            return y

@MODEL_REGISTRY.register('cifarresnext')
class Hendrycks2020AugMixResNeXtNet(CifarResNeXt):
    def __init__(self, depth=29, cardinality=4, base_width=32, class_number=100):
        super().__init__(ResNeXtBottleneck,
                         depth=depth,
                         class_number=class_number,
                         cardinality=cardinality,
                         base_width=base_width)
        # self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        # self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x, mode='compute-logit'):
        # x = (x - self.mu) / self.sigma
        return super().forward(x, mode)

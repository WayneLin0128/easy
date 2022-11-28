from utils import *
from args import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
# from torchvision.models import ResNet
#from se_module import SELayer

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.num_batches_tracked = 0
        self.se = SELayer(planes, 4)

        '''
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        '''
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )


    def forward(self, x):
        residual = x
        print("x before seblock size is: ", x.size())
        out = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.se(out)
        print("out size is: ", out.size())

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        '''
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if args.dropout > 0:
            out = F.dropout(out, p=args.dropout, training=self.training, inplace=True)
        '''
        out = F.dropout(out, p=args.dropout, training=self.training, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, feature_maps, input_shape, num_classes, few_shot, rotations):
        super(ResNet, self).__init__()
        # dropblock_size = 5
        self.inplanes = 3
        self.layer1 = self._make_layer(block, num_blocks[0], 64,
                                       stride=2)
        self.layer2 = self._make_layer(block, num_blocks[1], 160,
                                       stride=2)
        self.layer3 = self._make_layer(block, num_blocks[2], 320,
                                       stride=2)
        self.layer4 = self._make_layer(block, num_blocks[3], 640,
                                       stride=2)
        # self.keep_prob = 1
        # self.keep_avg_pool = False
        # self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        # self.drop_rate = args.dropout

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_classes = num_classes
        #self.classifier = nn.Linear(self.num_classes)
        self.linear = linear(640, num_classes)

        '''
        self.in_planes = feature_maps
        self.length = len(num_blocks)
        self.conv1 = nn.Conv2d(input_shape[0], feature_maps, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_maps)
        layers = []
        for i, nb in enumerate(num_blocks):
            layers.append(self._make_layer(block, (2 ** i) * feature_maps, nb, stride = 1 if i == 0 else 2))
        self.layers = nn.Sequential(*layers)
        self.linear = linear((2 ** (len(num_blocks) - 1)) * feature_maps, num_classes)
        self.linear_rot = linear((2 ** (len(num_blocks) - 1)) * feature_maps, 4)
        self.rotations = rotations
        self.depth = len(num_blocks)
        '''
    def _make_layer(self, block, n_block, planes, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample)
        else:
            layer = block(self.inplanes, planes, stride, downsample)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes)
            else:
                layer = block(self.inplanes, planes)
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x, index_mixup = None, lam = -1):
        x = self.layer1(x)
        f0 = x
        x = self.layer2(x)
        f1 = x
        x = self.layer3(x)
        f2 = x
        x = self.layer4(x)
        f3 = x
        x = x.view(x.size(0), -1)
        feat = x
        x = self.linear(x)
        return [f0, f1, f2, f3, feat], x

        '''
        if lam != -1:
            mixup_layer = random.randint(0, len(self.layers))
        else:
            mixup_layer = -1
        out = x
        if mixup_layer == 0:
            out = lam * out + (1 - lam) * out[index_mixup]
        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if mixup_layer == i + 1:
                out = lam * out + (1 - lam) * out[index_mixup]
            out = F.relu(out)
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        out = self.linear(features)
        if self.rotations:
            out_rot = self.linear_rot(features)
            return (out, out_rot), features
        return out, features
        '''

'''
def resnet12(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model
'''

def ResNet18(feature_maps, input_shape, num_classes, few_shot, rotations):
    model = ResNet(BasicBlock, [2, 2, 2, 2], feature_maps, input_shape, num_classes, few_shot, rotations)
    # model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def ResNet20(feature_maps, input_shape, num_classes, few_shot, rotations):
    return ResNet(BasicBlock, [3, 3, 3], feature_maps, input_shape, num_classes, few_shot, rotations)

from utils import *
from args import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class CaSE(nn.Module):
    def __init__(self, cin, reduction=64, min_units=16, standardize=True, out_mul=2.0, device=None, dtype=None):
        """
        Initialize a CaSE adaptive block.

        Parameters:
        cin (int): number of input channels.
        reduction (int): divider for computing number of hidden units.
        min_units (int): clip hidden units to this value (if lower).
        standardize (bool): standardize the input for the MLP.
        out_mul (float): multiply the MLP output by this value.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CaSE, self).__init__()
        self.cin = cin
        self.standardize = standardize
        self.out_mul = out_mul

        # Gamma-generator
        hidden_features = max(min_units, cin // reduction)
        self.gamma_generator = nn.Sequential(OrderedDict([
            ('gamma_lin1', nn.Linear(cin, hidden_features, bias=True, **factory_kwargs)),
            ('gamma_silu1', nn.SiLU()),
            ('gamma_lin2', nn.Linear(hidden_features, hidden_features, bias=True, **factory_kwargs)),
            ('gamma_silu2', nn.SiLU()),
            ('gamma_lin3', nn.Linear(hidden_features, cin, bias=True, **factory_kwargs)),
            ('gamma_sigmoid', nn.Sigmoid()),
        ]))

        self.gamma = torch.tensor([1.0])  # Set to one for the moment
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.gamma_generator.gamma_lin3.weight)
        torch.nn.init.zeros_(self.gamma_generator.gamma_lin3.bias)

    def forward(self, x):
        # Adaptive mode
        if (self.training):
            self.gamma = torch.mean(x, dim=[0, 2, 3])  # spatial + context pooling
            if (self.standardize):
                self.gamma = (self.gamma - torch.mean(self.gamma)) / torch.sqrt(
                    torch.var(self.gamma, unbiased=False) + 1e-5)
            self.gamma = self.gamma.unsqueeze(0)  # -> [1,channels]
            self.gamma = self.gamma_generator(self.gamma) * self.out_mul
            self.gamma = self.gamma.reshape([1, -1, 1, 1])
            return self.gamma * x  # Apply gamma to the input and return
        # Inference Mode
        else:
            self.gamma = self.gamma.to(x.device)
            return self.gamma * x  # Use previous gamma

    def extra_repr(self) -> str:
        return 'cin={}'.format(self.cin)

class BasicBlockRN12(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlockRN12, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.case1 = CaSE(cin=planes, reduction=64, min_units=16)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.case2 = CaSE(cin=planes, reduction=64, min_units=16)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.case3 = CaSE(cin=planes, reduction=64, min_units=16)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope = 0.1)
        out = self.case1(out)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope = 0.1)
        out = self.case2(out)
        out = self.bn3(self.conv3(out))
        out = self.case3(out)
        out += self.shortcut(x)
        if args.dropout > 0:
            out = F.dropout(out, p=args.dropout, training=self.training, inplace=True)
        return out
    
class ResNet12(nn.Module):
    def __init__(self, feature_maps, input_shape, num_classes, few_shot, rotations):
        super(ResNet12, self).__init__()        
        layers = []
        layers.append(BasicBlockRN12(input_shape[0], feature_maps))
        layers.append(BasicBlockRN12(feature_maps, int(2.5 * feature_maps)))
        layers.append(BasicBlockRN12(int(2.5 * feature_maps), 5 * feature_maps))
        layers.append(BasicBlockRN12(5 * feature_maps, 10 * feature_maps))        
        self.layers = nn.Sequential(*layers)
        self.linear = linear(10 * feature_maps, num_classes)
        self.rotations = rotations
        self.linear_rot = linear(10 * feature_maps, 4)
        self.mp = nn.MaxPool2d((2,2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, index_mixup = None, lam = -1):
        if lam != -1:
            mixup_layer = random.randint(0, 3)
        else:
            mixup_layer = -1
        out = x
        if mixup_layer == 0:
            out = lam * out + (1 - lam) * out[index_mixup]
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if mixup_layer == i + 1:
                out = lam * out + (1 - lam) * out[index_mixup]
            out = self.mp(F.leaky_relu(out, negative_slope = 0.1))
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        out = self.linear(features)
        if self.rotations:
            out_rot = self.linear_rot(features)
            return (out, out_rot), features
        return out, features

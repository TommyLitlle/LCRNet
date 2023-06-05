import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math

class Cluster_Conv(nn.Module):
    '''
    '''
    def __init__(self, in_channels, out_channels):
        super(Cluster_Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 64, kernel_size= 1, stride=1, padding=0, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, out_planes,
                                kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()
        self.GAP_1 = nn.AdaptiveAvgPool2d(1)
        self.GAP_3 = nn.AdaptiveAvgPool2d(2)
        self.GMP_1 = nn.AdaptiveMaxPool2d(1)
        self.GMP_3 = nn.AdaptiveMaxPool2d(2)
    
    def forward(self, x):
        N, C,_, _ = x.size()
        GAP_1 = self.GAP_1(x).view(N, C, -1)
        GAP_3 = self.GAP_3(x).view(N, C, -1)
        GMP_1 = self.GMP_1(x).view(N, C, -1)
        GMP_3 = self.GMP_3(x).view(N, C, -1)

        GAP_context = torch.cat((GAP_1, GAP_3), -1)
        GMP_context = torch.cat((GMP_1, GMP_3), -1)
        
        return GAP_context, GMP_context

class NSCA(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels):
        super(NSCA, self).__init__()
        self.in_channels = in_channels
        self.value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                                 kernel_size=1, stride=1, padding=0)

        self.psp = SPP()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.size()
      
        x_enhance = self.value(x)
        average_context, max_context = self.psp(x_enhance)

        max_context = max_context.permute(0, 2, 1)
        energy = torch.bmm(average_context, max_context)
        #print(energy.size())
        energy_new = torch.max(energy, -1, keepdim= True)[0].expand_as(energy)-energy
        #print(energy_new.size())
        attention = self.sigmoid(energy_new)
        x_value = x.view(N, C, -1)
        out = torch.bmm(attention, x_value)
        out = out.view(N, C, H, W)
        
        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0 \
            , dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size \
                , stride=stride, padding=padding, dilation=dilation \
                , groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01 \
                , affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)





def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)


        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)


        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers,  network_type, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR 
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)


        self.layer1 = self._make_layer(block, 64,  layers[0] )
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.ca = NSCA(512)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        #print(x.size())

        x = self.layer2(x)

        x = self.layer3(x)
        

        x = self.layer4(x)
        x= self.ca(x)
        if self.network_type == "ImageNet":
            x = self.avgpool(x)
            
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResidualNet(network_type, depth, num_classes):
    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes)

    return model


def anca_resnet18(pretrained=False, **kwargs):
    model = ResidualNet("ImageNet", 18, 7)
    return model

def anca_resnet34(pretrained=False, **kwargs):
    model = ResidualNet("ImageNet", 34, 7)
    return model

def anca_resnet50(pretrained=False, **kwargs):
    model = ResidualNet("ImageNet", 50, 7)
    return model

def anca_resnet101(pretrained=False, **kwargs):
    model = ResidualNet('ImageNet', 101, 1000)
    return model

if __name__ == '__main__':
    temp = torch.randn((1, 3, 224,224))
    net = anca_resnet18()
    net = net(temp)
    print(net.size())
   
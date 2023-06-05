"""SE-ResNet in PyTorch
Based on preact_resnet.py

Author: Xu Ma.
Date: Apr/15/2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#__all__ = ['SEResNet18', 'SEResNet34', 'SEResNet50', 'SEResNet101', 'SEResNet152','mylayer']

class SKFusion(nn.Module):
    
    def __init__(self, in_channels, out_channels, r = 16, L = 32, M=3):
        super(SKFusion, self).__init__()
        
        d = max(in_channels // r, L) 
        self.M = M
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.se=nn.Sequential(nn.Conv2d(in_channels,d,1,bias=False),
                               nn.ReLU(inplace=True))
        
        self.out_channels = out_channels
        
        
         # self 
        self.up_conv = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        self.center_conv = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        self.down_conv = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        
        self.fc2 = nn.Conv2d(d, out_channels, 1, 1, bias=False)
        
        self.softmax = nn.Softmax(dim=1) 
        
        
        
    def forward(self, x):
        
        batch_size,c, height, width = x.size()
        
        x1 = x[:, :, :height // 3,:] # up nuclear region
        x2 = x[:,:, height // 3: height*2 // 3,:]
        
        x3 = x[:, :, height*2 // 3: ,:]
        
        
        #up_y= self.up_conv(x1)
        #center_y = self.center_conv(x2)
        #down_y = self.down_conv(x3)
        
        up_y = self.avg_pool(x1)
        center_y = self.avg_pool(x2)
        down_y = self.avg_pool(x3)
         
        #y = torch.cat([up_y, center_y,  down_y],dim = 1 )
        y =  up_y + down_y + center_y
        
        
        #gap = self.global_pool(input)
        z = self.se(y)
        
        a_b_c = self.fc2(z)
        
        a_b_c = a_b_c.reshape(batch_size, self.M, self.out_channels // self.M, -1)
        a_b_c = self.softmax(a_b_c)
    
        #the part of selection
        a_b_c = list(a_b_c.chunk(self.M,dim=1)) #split to a b, and c 
        a_b_c = list(map(lambda x:x.reshape(batch_size,self.out_channels // self.M , 1, 1),a_b_c))
       
        a = a_b_c[0]
        b = a_b_c[1]
        c = a_b_c[2]


        up_region = a * x1
        center_region = b * x2
        down_region = c * x3
        
        
        y = torch.cat([up_region, center_region, down_region], dim = 2)
        #y = torch.squeeze(y, 2)
        #print(y.size())
          
        return y
    

    
    


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )




    def forward(self, x):
        
        b,c, height, width = x.size()
        x1 = x[:, :, :height // 2,:] # up nuclear region
        x2 = x[:,:, height // 2: ,:]
        up_y = self. avg_pool(x1)
        down_y = self.avg_pool(x2)
        whole_y = self.avg_pool(x)
        y = up_y + down_y + whole_y
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    
    


class SEPreActBlock(nn.Module):
    """SE pre-activation of the BasicBlock"""
    expansion = 1 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16):
        super(SEPreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.se = SKFusion(planes, planes*3)
        if stride !=1 or in_planes!=self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # Add SE block
        out = self.se(out)
        out += shortcut
        return out


class SEPreActBootleneck(nn.Module):
    """Pre-activation version of the bottleneck module"""
    expansion = 4 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16):
        super(SEPreActBootleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,self.expansion*planes,kernel_size=1,bias=False)
        self.se = SKFusion(planes, planes*3)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Add SE block
        out = self.se(out)
        out +=shortcut
        return out


class SEResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=3,reduction=16):
        super(SEResNet, self).__init__()
        self.in_planes=64
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,reduction=reduction)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,reduction=reduction)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,reduction=reduction)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,reduction=reduction)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.GAP = nn.AdaptiveAvgPool2d(1)

    #block means SEPreActBlock or SEPreActBootleneck
    def _make_layer(self,block, planes, num_blocks,stride,reduction):
        strides = [stride] + [1]*(num_blocks-1) # like [1,1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride,reduction))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.GAP(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def TZKResNet18(num_classes=3):
    return SEResNet(SEPreActBlock, [2,2,2,2],num_classes)


def TZKResNet34(num_classes=3):
    return SEResNet(SEPreActBlock, [3,4,6,3],num_classes)


def TZKResNet50(num_classes=3):
    return SEResNet(SEPreActBootleneck, [3,4,6,3],num_classes)


def ZKResNet101(num_classes=3):
    return SEResNet(SEPreActBootleneck, [3,4,23,3],num_classes)


def TZKResNet152(num_classes=3):
    return SEResNet(SEPreActBootleneck, [3,8,36,3],num_classes)


def test():
    net = TZKResNet18()
    y = net((torch.randn(2,3,32,32)))
    print(y.size())
if __name__ == '__main__':
  test()

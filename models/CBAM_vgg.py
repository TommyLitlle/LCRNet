'''VGG16 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from torchsummary import summary

#__all__=['SEVGG16']

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class VGGBlock(nn.Module):
    def __init__(self,in_channels, channels, stride=1):
        super(VGGBlock, self).__init__()
        self.conv =  nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.se = CBAM(channels)

    def forward(self, x):
        out=self.conv(x)
        out = F.relu(self.bn(out))
        out = self.se(out)
        return out

class CBAMVGG16(nn.Module):
    def __init__(self, num_classes=4,init_weights=True):
        super(CBAMVGG16, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.vggblock1 = self._make_layer(3,64,2)
        self.vggblock2 = self._make_layer(64, 128, 2)
        self.vggblock3 = self._make_layer(128, 256, 4)
        self.vggblock4 = self._make_layer(256, 512, 4)
        self.vggblock5 = self._make_layer(512, 512, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        if init_weights:
            self._initialize_weights()


    def _make_layer(self, in_channels,channels, num_blocks):

        layers = []
        layers.append(VGGBlock(in_channels, channels))
        for i in range(0,num_blocks-1):
            layers.append(VGGBlock(channels, channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m,'bias.data'):
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.vggblock1(x)
        out = self.maxpool(out)
        out = self.vggblock2(out)
        out = self.maxpool(out)
        out = self.vggblock3(out)
        out = self.maxpool(out)
        out = self.vggblock4(out)
        out = self.maxpool(out)
        out = self.vggblock5(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def test():
    net = CBAMVGG16(num_classes=3)
    x = torch.randn(1,3,224,112)
    y = net(x)
    print(y.size())
    # summary(net, input_size=(3, 32, 32))

if __name__=='__main__':
  test()






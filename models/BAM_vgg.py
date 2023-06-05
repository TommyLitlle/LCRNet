'''VGG16 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from torchsummary import summary

#__all__=['SEVGG16']

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        # self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        # avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        avg_pool = F.adaptive_avg_pool2d(in_tensor, 1)
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',    nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
                        padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)

class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor



class VGGBlock(nn.Module):
    def __init__(self,in_channels, channels, stride=1):
        super(VGGBlock, self).__init__()
        self.conv =  nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.se = BAM(channels)

    def forward(self, x):
        out=self.conv(x)
        out = F.relu(self.bn(out))
        out = self.se(out)
        return out

class BAMVGG16(nn.Module):
    def __init__(self, num_classes=4,init_weights=True):
        super(BAMVGG16, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.vggblock1 = self._make_layer(3,64,2)
        self.vggblock2 = self._make_layer(64, 128, 2)
        self.vggblock3 = self._make_layer(128, 256, 3)
        self.vggblock4 = self._make_layer(256, 512, 3)
        self.vggblock5 = self._make_layer(512, 512, 3)
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
    net = BAMVGG16(num_classes=3)
    x = torch.randn(2,3,224,112)
    y = net(x)
    print(y.size())
    # summary(net, input_size=(3, 32, 32))

if __name__=='__main__':
  test()






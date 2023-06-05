'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [32, 64, 'M', 128, 128, 'M', 192, 192, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 'M'],
}


    
class Mixed_Pooling(nn.Module):
    '''
    mixed pooling 
    '''
    def __init__(self):
        super(Mixed_Pooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.drop = nn.Dropout2d()
        
    def forward(self, x):
        GAP = self.gap(x)
        MP = self.mp(x)
        MG = torch.cat((GAP, MP), 1)
        
        # mixed 
        out = self.drop(MG)
        
        return out
        
        
    

    
    
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        
        self.classifier = nn.Linear(512, 3)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        
        #layers += [Mixed_Pooling()]
        layers += [nn.AdaptiveAvgPool2d(1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG19')
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.size())
    
if __name__ == '__main__':  
    test()

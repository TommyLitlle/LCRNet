
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import timm 


def MLP_mixer(num_classes=3):
    return timm.create_model('mixer_s16_224',num_classes=num_classes)

if __name__=='__main__':
    net = MLP_mixer()
    print(net)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import tensor
import torchvision
import torchvision.transforms as transforms

from timm.models.vision_transformer import vit_base_patch16_224_in21k as create_model

def Vit(num_classes=3):
    return create_model(num_classes=num_classes)
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import numpy as np
import random
import sys
import cv2


from utils import *
# from COT import *
from dataloader import *
import random
import copy
import yaml
from display_utils import *
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
import shutil
import sys
with open('config.yaml','rb') as f:
    config = yaml.safe_load(f.read())


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

exec('from models.' + config["model"]["file"] + ' import *')


seed = config["train"]["seed"]
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(12321)
# torch.use_deterministic_algorithms(True)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

batch_size = config["train"]["batch-size"]
base_learning_rate = config["train"]["lr"]
checkpoint_folder = config["model"]["folder"]
log_folder = config["log"]["folder"]
img_num = config["train"]["img_num"]
if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()
    # batch_size *= n_gpu
    base_learning_rate *= n_gpu

# Data (Default: CIFAR10)

def split_dataset(path):
    id_list = []
    for eye_id in os.listdir(path):
        p_id = eye_id.split("_")[0]
        if p_id not in id_list:
            id_list.append(p_id)
    random.shuffle(id_list)
    n = len(id_list)
    return id_list[:n*6//10], id_list[n*6//10:n*8//10], id_list[n*8//10:]



print('==> Preparing data.. (OCT3)')
# classes = 10
data_path = r'/data/home/zhangxiaoqing/USCD'
# data_path = r'/data/home/zhangxiaoqing/C_N_split_mix/all'
train_data, vaild_data, test_data = split_dataset(data_path)


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=config["train"]["mean"], std=config["train"]["std"])
])

transform_vaild = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=config["train"]["mean"], std=config["train"]["std"])
])

trainset = dataset(os.path.join(data_path, 'training'), transform=transform_train)
trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=8,drop_last=True)
vaildset = dataset(os.path.join(data_path, 'validation'), transform=transform_vaild)
vaildset_add = dataset(os.path.join(data_path, 'validation'), transform=transform_train)

vaildloader = torch.utils.data.DataLoader(
    vaildset, batch_size=batch_size, shuffle=False, num_workers=8)
vaildloader_add = torch.utils.data.DataLoader(
    vaildset_add, batch_size=batch_size, shuffle=False, num_workers=8)

# Model

model_name = config["model"]["name"]
resume = config["train"]["resume"]
logname = os.path.join(sys.path[0], log_folder, model_name + config["train"]["sess"] + '_' + str(seed) + '.csv')

if not os.path.exists(os.path.join(sys.path[0], log_folder)):
    os.makedirs(os.path.join(sys.path[0], log_folder))

shutil.copyfile(os.path.join(sys.path[0], 'config.yaml'), os.path.join(sys.path[0], log_folder, model_name + config["train"]["sess"] + '_' + str(seed) + '_log.log'))



if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(os.path.join(sys.path[0], checkpoint_folder)), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(sys.path[0], checkpoint_folder, model_name +
                                   config["train"]["sess"] + '_' + str(seed)))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
else:
    print('==> Building model.. (Default : PreActResNet18)')
    start_epoch = 0
    # net = PreActResNet18_CIFAR100()
    net=eval(config["model"]["net"])

print(net)
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    # cudnn.benchmark = True
    cudnn.deterministic = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=base_learning_rate,
                      momentum=0.9, weight_decay=config["train"]["decay"])

# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
           
        # Baseline Implementation
        inputs, targets = Variable(inputs), Variable(targets)
        #############################################################################
        # feature = []
        # y = inputs
       
        # for name, module in net.module._modules.items():
        #     if name == "fc":
        #         y = y.squeeze()
        #     if name == "layer1":
        #         k = 0
        #         for block in module.children():
        #             k += 1
        #             c = 0
        #             o = y
        #             for layer in block.children():
        #                 y = layer(y)
        #                 if c == 4:
        #                     N, C, _, _ = y.size()
        #                     feature.append(y.permute(1, 0, 2, 3).contiguous().view(C, -1).cpu())
        #                     map_save_path = os.path.join(sys.path[0],"maps", "epoch_{}".format(str(epoch)), "layer1_{}_maps".format(str(k)))
        #                     if not os.path.exists(map_save_path):
        #                         os.makedirs(map_save_path)
        #                     for i in range(N):
        #                         map = standardize_and_clip(torch.tensor(y.mean(1).cpu())[i]).numpy()
        #                         map = cv2.applyColorMap(np.uint8(255 * map), cv2.COLORMAP_JET)
        #                         cv2.imwrite(os.path.join(map_save_path, img_name[i]), map)
        #                 if c == 5:
        #                     y = nn.ReLU()(y + o)
        #                 c += 1
        #     else:
        #         y = module(y)
        # outputs = y
        #print(inputs.size())
        outputs = net(inputs)
        #############################################################################
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return (train_loss / batch_idx, 100. * correct / total)


def vaild(epoch):
    global best_acc
    net.eval()
    vaild_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(vaildloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
      
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            vaild_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(batch_idx, len(vaildloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (vaild_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        for batch_idx, (inputs, targets) in enumerate(vaildloader_add):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            vaild_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(batch_idx, len(vaildloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (vaild_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch)
    return (vaild_loss / batch_idx, 100. * correct / total)

def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir(os.path.join(sys.path[0], 'checkpoint')):
        os.mkdir(os.path.join(sys.path[0], 'checkpoint'))
    torch.save(state, os.path.join(sys.path[0], 'checkpoint', model_name +
                                   config["train"]["sess"] + '_' + str(seed)))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""

    lr = base_learning_rate
    if epoch <= 9:
        # warm-up training for large minibatch
        lr = base_learning_rate + base_learning_rate* epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(
            ['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

for epoch in range(start_epoch, config["train"]["epochs"]):
    adjust_learning_rate(optimizer, epoch)
    # complement_adjust_learning_rate(complement_optimizer, epoch)
    train_loss, train_acc = train(epoch)
    vaild_loss, vaild_acc = vaild(epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc, vaild_loss, vaild_acc])


def trans_classReport(data):
    data = data.split()
    x = data.index('accuracy')
    n = (x + 1)//5-1
    tag = data[:4]
    m = {}
    for i in range(n):
        start = data.index(str(i)) + 1
        for j in range(4):
            m["grade"+str(i)+"_"+tag[j]] = data[start+j]
    m[data[x]] = data[x + 1]
    m_x = data.index("macro")
    for i in range(3):
        m[data[m_x] + "_" + tag[i]] = data[m_x + 2 + i]
    w_x = data.index("weighted")
    for i in range(3):
        m[data[w_x] + "_" + tag[i]] = data[w_x + 2 + i]
    m["total"] = data[-1]
    return m
# Testing



def test():
    print("begin")
    global best_acc
    global keys
    model_path = os.path.join(sys.path[0], 'checkpoint', model_name + config["train"]["sess"] + '_' + str(seed))
    print(model_path)
    checkpoint = torch.load(model_path)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    epoch = checkpoint['epoch']
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["train"]["mean"], std=config["train"]["std"])
    ])

    testset = dataset(os.path.join(data_path, 'validation'), transform=transform_vaild)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8)

    # torch.set_rng_state(checkpoint['rng_state'])
    if use_cuda:
        net.cuda()
        # net = torch.nn.DataParallel(net)
        print('Using', torch.cuda.device_count(), 'GPUs.')
        # cudnn.benchmark = True
        cudnn.deterministic = True
        print('Using CUDA..')
    net.eval()
    print(epoch)
    vaild_loss = 0
    correct = 0
    total = 0
    labels = []
    pre_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            vaild_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            labels.extend(targets.data.cpu())
            pre_labels.extend(predicted.data.cpu())
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (vaild_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    labels = np.reshape(labels, [-1, 1])
    pre_labels = np.reshape(pre_labels, [-1, 1])
    m = trans_classReport(classification_report(labels, pre_labels, digits=5))
    m['model_name'] = os.path.basename(model_path)
    m['epoch'] = epoch
    m['vaild_acc'] = best_acc
    m['balanced_acc'] = balanced_accuracy_score(labels, pre_labels)
    m['kappa'] = cohen_kappa_score(labels, pre_labels)
    # data = ",".join(str(i) for i in data)
    # print(data, file=f)
    with open(logname, 'a') as f:
        f.write(classification_report(labels, pre_labels, digits=5))
        f.write("epoch: ")
        f.write(str(epoch))
        f.write("\n")
        f.write("vaild_acc: ")
        f.write(str(best_acc))
        f.write("\n")
        f.write("kappa: ")
        f.write(str(cohen_kappa_score(labels, pre_labels)))


#train(config["train"]["epochs"])
if __name__ == '__main__':
    test()



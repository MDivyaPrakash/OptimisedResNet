#This file has the implementation of RESNET 18

import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
import math
#here we are using Arch 2,
from model2 import *
from utils import progress_bar
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
# Training
def train(epochs,criterion,optimizer,scheduler,net):
    best_acc_es=0
    best_test=0
    counter=0
    best_acc=0
    prev_loss=math.inf
    for i in tqdm(range(epochs)):
        print('\nEpoch: %d' % (i+1))
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc=100.*correct/total
        print('-'*10)
        print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('-'*10)
        if prev_loss>(train_loss/(batch_idx+1)):
            counter=0
            prev_loss=(train_loss/(batch_idx+1))
            if acc > best_acc:
                best_acc=acc
                print("Validating!")
                net.eval()
                test_loss = 0
                correct = 0
                total = 0
                for batch_idx, (inputs, targets) in enumerate(valloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                test_acc=100.*correct/total
                print('-'*10)
                print('Val Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                print('-'*10)
                if best_test<test_acc:
                    best_test=test_acc
                    print("Saving!")
                    torch.save(net,'./es9-dl-model2.pt')
        else:
            counter+=1
            prev_loss=(train_loss/(batch_idx+1))
        if counter > 5:
            break

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        best_acc = acc
        torch.save(net,'./arch1-es.pt')
    return all_predicts, all_targets
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    print('Setting Seed')
    seed_everything(77)
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    train_idx, val_idx = train_test_split(range(len(trainset)), test_size=0.2)
    train_dataset = Subset(trainset, train_idx)
    val_dataset = Subset(valset, val_idx)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')

    # Model

    layer_structure = [3,3,3]
    print('==> Building model.. with layer structure :',layer_structure)
    net = ResNet18(layer_structure)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    print(net)
    learning_rate = 0.1
    print("==> Learning Rate " ,learning_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("The number of trainable parameters is: " ,pytorch_total_params)

    print('==> Model training and testing with the layer structure: ',layer_structure)
    train(200,criterion,optimizer,scheduler,net)

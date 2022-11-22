#This file has the modified implementation of RESNET 18

import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnetModel import *
from utils import progress_bar

import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

plot_train_accuracy,plot_val_accuracy = [],[]
plot_train_loss,plot_val_loss = [],[]

# Training function
def train(epoch):
    print('\nEpoch: %d' % (epoch+1))
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

        progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Train Accuracy: %.3f%%'
                     % (train_loss/(batch_idx+1), 100.*correct/total))
    plot_train_loss.append((train_loss/(batch_idx+1)))
    plot_train_accuracy.append((100.*correct/total))
    
#Validation Function 
def validate(epoch):
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_predicts=[]
    all_targets=[]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader), 'Validation Loss: %.3f | Validation Accuracy: %.3f%%'
                         % (val_loss/(batch_idx+1), 100.*correct/total))

    # Save checkpoint.
    plot_val_loss.append((val_loss/(batch_idx+1)))
    plot_val_accuracy.append((100.*correct/total))
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_FinalResult4_ConfusionMatrix_DP2e-1_SGD.pth')
        best_acc = acc

# Testing function
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_predicts=[]
    all_targets=[]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Test Acc: %.3f%%'
                         % (test_loss/(batch_idx+1), 100.*correct/total))
            y=predicted.cpu().numpy().tolist()
            y_true=targets.cpu().numpy().tolist()
            for i in y:
                all_predicts.append(i)
            for k in y_true:
                all_targets.append(k)

    return all_predicts, all_targets

        
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data Augmentation
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
    # splitting of Training Set to train and validation
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

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt_FinalResult4_ConfusionMatrix_DP2e-1_SGD.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    learning_rate = 0.1
    print("==> Learning Rate " ,learning_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    print("==> Optimizer " ,optimizer)
    print("==> Scheduler " ,scheduler)
    

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("The number of trainable parameters is: " ,pytorch_total_params)

    print('==> Model training and testing with the layer structure: ',layer_structure)

    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        validate(epoch)
        scheduler.step()
    print(f'Best Accuracy: {best_acc} in Epochs: {epoch+1}')
        
    #Confusion Matrix
    print("==> Generating Confusion Matrix for the best model")
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_FinalResult4_ConfusionMatrix_DP2e-1_SGD.pth')
    net.load_state_dict(checkpoint['net'])
                                  
    pred,real = test(epoch)
    cm = confusion_matrix(real, pred)
    print(cm)
                                  
    print("==> Saving Variables for the plot")
    with open('./Confusion_Matrix/Plotckpt_FinalResult4_ConfusionMatrix_DP2e-1_SGD.npy', 'wb') as f:
        np.save(f,cm) 
        np.save(f,plot_train_accuracy)
        np.save(f,plot_val_accuracy)
        np.save(f,plot_train_loss)  
        np.save(f,plot_val_loss)
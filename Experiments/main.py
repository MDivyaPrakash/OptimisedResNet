#This file has the implementation of RESNET 18

import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune

import torchvision
import torchvision.transforms as transforms

import os
import argparse

#from models import *
from resnetModel import *
from utils import progress_bar

# Training
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
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
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--optimizer','-o',default='SGD',action='store',help='Choose optimizer: sgd, adam, sgdnest, adag, adad')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout_rate')
    parser.add_argument('--numwork','-nw',type=int, default=2, action='store',help='Number of workers to use')
    parser.add_argument('--epochs','-ep',type=int, default=200, action='store',help='Number of epochs')
    parser.add_argument('--lr','-lr',type=float, default=0.01,action='store',help='Learning rate')
    parser.add_argument('--tmax','-tm',type=int, default=200,action='store',help='T - max')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Device is set to " + device)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=args.numwork)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.numwork)
		
    print(" Number of workers is " + str(args.numwork))
    print(" Number of epochs is " + str(args.epochs))
    print(" Learning rate is " + str(args.lr))

    classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')

    # Model

    layer_structure = [2,2,2,2]
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
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    learning_rate = args.lr
    print("==> Learning Rate " ,learning_rate)
    criterion = nn.CrossEntropyLoss()
    chosen_optimizer = args.optimizer

    if chosen_optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax)
        print("******************* Chosen optimizer is " + chosen_optimizer + " *******************")
    elif chosen_optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax)
        print("******************* Chosen optimizer is " + chosen_optimizer + " *******************")
    elif chosen_optimizer == 'sgdnest':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax)
        print("******************* Chosen optimizer is " + chosen_optimizer + " *******************")
    elif chosen_optimizer == 'adad':
        optimizer = optim.Adadelta(net.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax)
        print("******************* Chosen optimizer is " + chosen_optimizer + " *******************")
    elif chosen_optimizer == 'adag':
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax)
        print("******************* Chosen optimizer is " + chosen_optimizer + " *******************")
    else:
        print("*********** Default optimzer: SGD chosen ***********")
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax)

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("The number of trainable parameters is: " ,pytorch_total_params)

    print('==> Model training and testing with the layer structure: ',layer_structure)
    for epoch in range(start_epoch, start_epoch+args.epochs):

        train(epoch)
        test(epoch)
        scheduler.step()
    #### Pruning
    #print(list(net.named_parameters()))
    #print(net.parameters())
    #print("==>Global Pruning operation")
    #pruned_net = copy.deepcopy(net)
    #parameters_to_prune = []
    #for module_name, module in pruned_net.named_modules():
    #    if isinstance(module, torch.nn.Conv2d):
    #        parameters_to_prune.append((module, "weight"))
    #prune.global_unstructured(
    #    parameters_to_prune,
    #    pruning_method=prune.L1Unstructured,
    #    amount=0.57,
    #)

    #remove_parameters(model = pruned_net)
    #print(len(parameters_to_prune))
    #pytorch_total_params_ap = sum(p.numel() for p in pruned_net.parameters() if p.requires_grad)
    #print("The number of trainable parameters after pruning is: " ,pytorch_total_params_ap)
    #Testing the accuracy on test data
    #test(1)



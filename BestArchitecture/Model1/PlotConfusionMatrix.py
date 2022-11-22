#This file contains the plotting of the Test vs Validiation Losses/Accuracies and also the Confusion matix
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import os
import argparse

import numpy as np
CLASSES = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#Confusion Matrix
with open('./Confusion_Matrix/Plotckpt_FinalResult1_DP0_SGD.npy', 'rb') as f:
    print("==> Confusion Matrix")
    cm = np.load(f)
    display = ConfusionMatrixDisplay(cm,display_labels = CLASSES).plot()
    plt.show()
    
    print("==> Plot variables")
    
    plot_train_accuracy = np.load(f)
    plot_val_accuracy = np.load(f)
    plot_train_loss = np.load(f)
    plot_val_loss = np.load(f)
    
    epochs = range(1,201)
    plt.plot(epochs, plot_train_accuracy, 'g', label='Training Accuracy')
    plt.plot(epochs, plot_val_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    plt.plot(epochs, plot_train_loss, 'g', label='Training Loss')
    plt.plot(epochs, plot_val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
import datasets
from utils.trainer import Trainer
from utils.helper import Save_Handle
import logging
import os
import sys
import time
import torch
from torchvision import transforms
from torch import optim
from torch import nn

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


Dataset = getattr(datasets, 'Shoes')
datasets = {x: Dataset(os.path.join('/home/teddy/shoes', x), data_transforms[x])
                 for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=128,
                                              shuffle=(True if x == 'train' else False),
                                                num_workers=8, pin_memory=True)
                    for x in ['train', 'val']}

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2' # set vis gpu
device = torch.device("cuda")
for inputs, labels in dataloaders['train']:
    labels = labels.to(device)
    print(labels)
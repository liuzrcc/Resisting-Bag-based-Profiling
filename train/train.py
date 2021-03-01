from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from random import shuffle


import time
import os
import copy

import pandas as pd

plt.ion()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 0.0001
num_epochs = 25
batch_size = 128
net = 'alexnet'
personality_to_train = 'C'
percent_train = '100p'

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        # transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


users_list = sorted(os.listdir('./data/PF_all/'))
data_loaded = np.load('./data/' + personality_to_train + '.npy', allow_pickle=True)
img_root = './data/PF_all/'

train_ls = []
for item in data_loaded.item()['train_low']:
    img_dir = img_root + users_list[item] + '/'
    for img in os.listdir(img_dir):
        train_ls = train_ls+ [[img_dir + img, 0]]

for item in data_loaded.item()['train_high']:
    img_dir = img_root + users_list[item] + '/'
    for img in os.listdir(img_dir):
        train_ls = train_ls+ [[img_dir + img, 1]]
shuffle(train_ls) 

val_ls = []
for item in data_loaded.item()['val_low']:
    img_dir = img_root + users_list[item] + '/'
    for img in os.listdir(img_dir):
        val_ls = val_ls+ [[img_dir + img, 0]]

for item in data_loaded.item()['val_high']:
    img_dir = img_root + users_list[item] + '/'
    for img in os.listdir(img_dir):
        val_ls = val_ls+ [[img_dir + img, 1]]


test_ls = []
for item in data_loaded.item()['test_low']:
    img_dir = img_root + users_list[item] + '/'
    for img in os.listdir(img_dir):
        test_ls = test_ls+ [[img_dir + img, 0]]

for item in data_loaded.item()['test_high']:
    img_dir = img_root + users_list[item] + '/'
    for img in os.listdir(img_dir):
        test_ls = test_ls+ [[img_dir + img, 1]]


class trainset(Dataset):
    def __init__(self, loaded_ls, phrase, percent_train):
        if percent_train == '100p':
            self.target = loaded_ls
            self.phrase = phrase
        elif percent_train == '50p':
            self.target = loaded_ls[:int(0.5*len(loaded_ls))]
            self.phrase = phrase
        
    def __getitem__(self, index):
        target = data_transforms[self.phrase](Image.open(self.target[index][0]).convert('RGB'))
        label = self.target[index][1]
        return target, label

    def __len__(self):
        return len(self.target)


image_datasets = {'train':trainset(train_ls, 'train', percent_train), 'val': trainset(val_ls, 'val', '100p'), 'test':trainset(test_ls, 'test', '100p')}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                            shuffle=True, num_workers=8)
            for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

print(dataset_sizes['train'])

writer = SummaryWriter('runs/' + personality_to_train, comment = 'A_vgg_lr_' + str(lr))

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                writer.add_scalar('training loss', running_loss / dataset_sizes[phase], epoch)
                scheduler.step()

            

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'val':
                writer.add_scalar('val loss', running_loss / dataset_sizes[phase], epoch)
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    writer.close()
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if net == 'vgg':
    model_ft = models.vgg16(pretrained=True)

    for param in model_ft.parameters():
        param.requires_grad = False

    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,2)

elif net == 'alexnet':
    model_ft = models.alexnet(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False
        
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.classifier.parameters(), lr=lr, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)



model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=num_epochs)

def test_model(model):
    running_corrects = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
    

        # statistics
    
        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / dataset_sizes['test']
    print('test Acc: {:4f}'.format(epoch_acc))
    return epoch_acc

test_model(model_ft.eval())

state_dict = model_ft.state_dict()
torch.save(state_dict, './models/' + personality_to_train + '_' + net + '_' + percent_train + '.pth')

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

import time
import os
import copy
from tqdm import tqdm_notebook 

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

plt.ion()
device = 'cuda:0'
users_list = sorted(os.listdir('./data/PF_all/'))



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'adv_val': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
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

class load_from_list(Dataset):
    def __init__(self, train_ls):
        self.target = train_ls

    def __getitem__(self, index):
        target = data_transforms['val'](Image.open(self.target[index]).convert('RGB'))
        return target

    def __len__(self):
        return len(self.target)

def sigmoid(z):
    return 1/(1 + torch.exp(-z))

def ddl_direct(f_0, personality_trait, net):
    centroid = np.load('./data/DD_param/' + net + '_' + personality_trait + '_dd.npy', allow_pickle=True)
    c_pos = torch.tensor(centroid.item()['cen_pos']).to(device)
    c_neg = torch.tensor(centroid.item()['cen_neg']).to(device)
    d_mean = torch.tensor(centroid.item()['mean']).to(device)
    d_std = torch.tensor(centroid.item()['std']).to(device)

    cos_torch = nn.CosineSimilarity(dim=1, eps=1e-6)
    d_max = torch.max(torch.cat((cos_torch(f_0, c_neg).reshape(-1, 1), cos_torch(f_0, c_pos).reshape(-1, 1)), axis = 1), axis = 1) 
    d_min = torch.min(torch.cat((cos_torch(f_0, c_neg).reshape(-1, 1), cos_torch(f_0, c_pos).reshape(-1, 1)), axis = 1), axis = 1)
    D_i = d_max.values / d_min.values
    D_i = sigmoid((D_i - d_mean) / d_std)
    return D_i


def sample_smaller_test(data_dir, pos_user, neg_user, normal_sample_size = 25, adv_size = 5, user_num = 20):
    res = np.empty([0, 4])
    np.random.seed(1234)
    for user in pos_user:
        ls = os.listdir(data_dir + users_list[user] + '/')
        for i in range(int(user_num / 2)):
            rand_ls = np.array([ls[j] for j in np.random.randint(200, size=normal_sample_size)])
            adv_ls = np.array([ls[j] for j in np.random.randint(200, size=adv_size)])
            res = np.append(res, np.array([[user, 1, rand_ls, adv_ls]]), axis = 0)
   
    for user in neg_user:
        ls = os.listdir(data_dir+ users_list[user] + '/')
        for i in range(int(user_num / 2)):
            rand_ls = np.array([ls[j] for j in np.random.randint(200, size=normal_sample_size)])
            adv_ls = np.array([ls[j] for j in np.random.randint(200, size=adv_size)])
            res = np.append(res, np.array([[user, 0, rand_ls, adv_ls]]), axis = 0)

    return res


Threshold_dict = {
'A':{'alexnet_strong_A':{11:0.7, 15:0.7, 25:0.8},'alexnet_weak_A':{11:0.1, 15:0.2, 25:0.1},
'vgg_strong_A': {11:0.5, 15:0.5, 25: 0.3},'vgg_weak_A': {11:0.8 , 15:0.9, 25: 0.9}},
'O':{'alexnet_strong_O':{11:0.9, 15:0.9, 25:0.9},'alexnet_weak_O':{11:0.1, 15:0.6, 25:0.4},
'vgg_strong_O':{11:0.6, 15:0.6, 25:0.7}, 'vgg_weak_O':{11:0.3, 15:0.1, 25:0.3}},
'C':{'alexnet_strong_C':{11:0.9, 15:0.8, 25:0.8},'alexnet_weak_C':{11:0.5, 15:0.1, 25:0.6},
'vgg_strong_C':{11:0.6, 15:0.7, 25:0.5}, 'vgg_weak_C':{11:0.6, 15:0.2, 25:0.1}},
'E':{'alexnet_strong_E':{11:0.9, 15:0.6, 25:0.6},'alexnet_weak_E':{11:0.1, 15:0.3, 25:0.5},
'vgg_strong_E':{11:0.2 , 15:0.5, 25: 0.6}, 'vgg_weak_E':{11:0.2 , 15:0.2, 25: 0.2}},
'N':{'alexnet_strong_N':{11:0.9, 15:0.9, 25:0.6},'alexnet_weak_N':{11:0.5, 15:0.2, 25:0.3},
'vgg_strong_N':{11:0.8 , 15:0.8, 25: 0.1}, 'vgg_weak_N':{11:0.9 , 15:0.8, 25: 0.8}}}

Normal_Threshold_dict = {
'A':{'alexnet_strong_A':0.8,'alexnet_weak_A':0.5,
'vgg_strong_A': 0.4,'vgg_weak_A': 0.4},
'O':{'alexnet_strong_O':0.9,'alexnet_weak_O':0.5,
'vgg_strong_O':0.5, 'vgg_weak_O':0.5},
'C':{'alexnet_strong_C':0.9,'alexnet_weak_C':0.6,
'vgg_strong_C':0.2, 'vgg_weak_C':0.9},
'E':{'alexnet_strong_E':0.7,'alexnet_weak_E':0.7,
'vgg_strong_E':0.5, 'vgg_weak_E':0.1},
'N':{'alexnet_strong_N':0.9,'alexnet_weak_N':0.9,
'vgg_strong_N':0.5, 'vgg_weak_N':0.9}}   

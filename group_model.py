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

from utils import data_transforms, ddl_direct, sample_smaller_test, Threshold_dict, load_from_list

plt.ion()
device = 'cuda:0'


def model_select_MV(net, personality_trait):
    if net == 'vgg':
        model_ft = models.vgg16(pretrained=True)
    elif net == 'alexnet':
        model_ft = models.alexnet(pretrained=True)

    for param in model_ft.parameters():
        param.requires_grad = False

    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,2)

    model_ft.load_state_dict(torch.load('./models/' + personality_trait + '_' + net +'_' + '100p' +'.pth'), strict=False)
    model_ft.eval().cuda()
    return model_ft

def model_select_fusion(net, personality_trait):
    if net == 'vgg':
        model = models.vgg16(pretrained=True)
    elif net == 'alexnet':
        model = models.alexnet(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs,2)

    model.load_state_dict(torch.load('./models/' + personality_trait + '_' + net +'_100p.pth'), strict=False)
    model.features = model.features[:]

    final_classifier = model.classifier[4:]

    model.classifier = model.classifier[:4]

    final_classifier.eval().cuda()
    model = model.eval().cuda()
    return model, final_classifier

def MV(temp_ls, t_test, u_idx, model_ft):
    dd_weights = np.empty((0))
    for data in DataLoader(load_from_list(temp_ls), batch_size = 40, shuffle = False, num_workers = 4):
        res = model_ft(data.cuda()).cpu().detach().argmax(1).numpy().reshape((1, -1))
        dd_weights = np.append(dd_weights, res)
    
#     for idx, img in enumerate(temp_ls):
#         dd_weights[idx] = model_ft(data_transforms['val'](Image.open(img).convert('RGB')).unsqueeze(0).cuda()).reshape((1, -1)).cpu().detach().argmax(1)
    ct = (dd_weights == t_test[u_idx][1]).sum()
    # print(ct)
    # print(0.5 * len(dd_weights))
    label = (ct > (0.5 * len(dd_weights)))
    roc_ls =  [(ct > (0.5 * len(dd_weights))), t_test[u_idx][1]]
    return label, roc_ls


def weighted(temp_ls, t_test, u_idx, model, final_classifier, personality_trait, threshold, net, average = False):
    if not average:
        for data in DataLoader(load_from_list(temp_ls), batch_size = 40, shuffle = False, num_workers = 4):
            dd_weights = ddl_direct(model(data.cuda()), personality_trait, net).cpu().detach()
            group_feature = model(data.cuda()).cpu().detach()


        K = (1/ (dd_weights.max() - dd_weights.min()))
        R = K * dd_weights  + (1 - K * torch.max(dd_weights))
        idx = (R > threshold)
        group_rep = torch.mean((R[idx].reshape((-1, 1)) * group_feature[torch.where(idx == True)[0]]), axis =0)
        label = (final_classifier(group_rep.cuda().float()).argmax(0) == t_test[u_idx][1]).cpu().float().numpy()
        roc_ls = [nn.Softmax()(final_classifier(group_rep.cuda().float()))[1].detach().cpu().numpy(), t_test[u_idx][1]]
    else:
        for data in DataLoader(load_from_list(temp_ls), batch_size = 40, shuffle = False, num_workers = 4):
            group_feature = model(data.cuda()).detach()
        
        group_rep = torch.mean(group_feature, axis=0)
        
        label = (final_classifier(group_rep.cuda().float()).argmax(0) == t_test[u_idx][1]).cpu().float().numpy()
        roc_ls = [nn.Softmax()(final_classifier(group_rep.cuda().float()))[1].detach().cpu().numpy(), t_test[u_idx][1]]

    return label, roc_ls


def model_select_pretrained(net):
    if net == 'vgg':
        model = models.vgg16(pretrained=True)
    elif net == 'alexnet':
        model = models.alexnet(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs,2)

    model.features = model.features[:]
    final_classifier = model.classifier[4:]

    model.classifier = model.classifier[:4]

    final_classifier.eval().cuda()
    model = model.eval().cuda()
    return model, final_classifier

def model_select_pretrained_resnet():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model = nn.Sequential(*list(model.children())[:-1])
    return model
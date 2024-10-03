import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

# from pf.new_PF_bags_loader import PFBags
from model import Attention, GatedAttention, SetTransformer, DeepSet
from dataloader import E6instance
from torchvision import transforms, models


data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



def pgd_linf_pire(resnet50, X, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):

        loss = torch.sqrt(nn.MSELoss()(resnet50(X), resnet50(delta)))
        # print(loss.item())
        loss.backward(retain_graph=True)
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()


def pgd_linf_pire_advimage(resnet50, X, images, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):

        loss = torch.sqrt(nn.MSELoss()(resnet50(X), resnet50(images + delta)))
        # print(loss.item())
        loss.backward(retain_graph=True)
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

from __future__ import print_function
import sys

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloader import *
from model import Attention, GatedAttention, SetTransformer, DeepSet, ResNet18, Attention224
from util import setup_logger
import os
from dataloader import E6instance, E6feature

import torch.nn as nn
import torchvision.models as models

# Load the pre-trained ResNet-18 model


# Training settings
parser = argparse.ArgumentParser(description='att MIL')
parser.add_argument('--exp_path', default='exp/example/', help='exp_path')
parser.add_argument('--method', default='attmil', help='attmil, deepset, sett, svit')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-4, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')
parser.add_argument('--Emocat', type=str, default='anger', help='')


# logger setup
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

log_file_path = os.path.join(args.exp_path)
if not os.path.exists(log_file_path):
    os.mkdir(log_file_path)
logger = setup_logger(name='att mil', log_file=log_file_path +'/' + args.Emocat + ".log")

logger.info("PyTorch Version: %s" % (torch.__version__))
logger.info("Experiment: %s" % (torch.__version__))
logger.info("Expetiment Path: %s" % (args.exp_path))
logger.info("Method is: %s" % (args.method))


data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
vgg16.classifier = vgg16.classifier[:-1]
vgg16.cuda()
vgg16.eval()
pass



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train, Val, and Test Set')
loader_kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

if args.method == 'attmil':

    train_loader = data_utils.DataLoader(E6instance('train', args.Emocat),
                                        batch_size=1,
                                        shuffle=True,
                                        **loader_kwargs)

    val_loader= data_utils.DataLoader(E6instance('val', args.Emocat),
                                        batch_size=1,
                                        shuffle=True,
                                        **loader_kwargs)

    test_loader= data_utils.DataLoader(E6instance('test', args.Emocat),
                                        batch_size=1,
                                        shuffle=False,
                                        **loader_kwargs)

    print('Init Model')
    if args.model=='attention':
        model = Attention()
    elif args.model=='gated_attention':
        model = GatedAttention()
    if args.cuda:
        model.cuda()

elif args.method == 'sett':

    train_loader = data_utils.DataLoader(E6instance224('train', args.Emocat),
                                        batch_size=1,
                                        shuffle=True,
                                        **loader_kwargs)
    val_loader= data_utils.DataLoader(E6instance224('val', args.Emocat),
                                        batch_size=1,
                                        shuffle=True,
                                        **loader_kwargs)

    test_loader= data_utils.DataLoader(E6instance224('test', args.Emocat),
                                        batch_size=1,
                                        shuffle=False,
                                        **loader_kwargs)


    model = SetTransformer()
    model.cuda()


elif args.method == 'deepset':

    train_loader = data_utils.DataLoader(E6instance224('train', args.Emocat),
                                        batch_size=1,
                                        shuffle=True,
                                        **loader_kwargs)
    val_loader= data_utils.DataLoader(E6instance224('val', args.Emocat),
                                        batch_size=1,
                                        shuffle=True,
                                        **loader_kwargs)

    test_loader= data_utils.DataLoader(E6instance224('test', args.Emocat),
                                        batch_size=1,
                                        shuffle=False,
                                        **loader_kwargs)


  
    model = DeepSet(dim_input=4096,num_outputs=1, dim_output=2)
    model.cuda()


elif args.method == 'cnnvote':

    train_loader = data_utils.DataLoader(E6singleimage('train', args.Emocat),
                                        batch_size=64,
                                        shuffle=True,
                                        **loader_kwargs)
    val_loader= data_utils.DataLoader(E6singleimage('val', args.Emocat),
                                        batch_size=64,
                                        shuffle=True,
                                        **loader_kwargs)

    test_loader= data_utils.DataLoader(E6instance('test', args.Emocat),
                                        batch_size=1,
                                        shuffle=False,
                                        **loader_kwargs)

    model = models.resnet18(pretrained=True)
    # model = ResNet18(2)
    # for param in model.parameters():
    #     param.requires_grad = False
    
    num_classes = 2
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.cuda()

else:
    raise NotImplementedError



optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
if args.method == 'sett' or args.method == 'deepset' or args.method == 'cnnvote':
    criteria  = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    # global loss

    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        # data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        if args.method == 'attmil':
            loss, _ = model.calculate_objective(data, bag_label)
            train_loss += loss.data[0].cpu().numpy()[0]
        elif args.method == 'sett':
            pred = model(vgg16(data[0]).unsqueeze(0)).unsqueeze(0)
            loss = criteria(pred, bag_label.to(torch.uint8))
            train_loss += loss.item()
        elif  args.method == 'deepset':
            pred = model(vgg16(data[0]).unsqueeze(0))
            loss = criteria(pred[0], bag_label.to(torch.uint8))
            train_loss += loss.item()
        elif  args.method == 'cnnvote':
            pred = model(data[:, 0])
            loss = criteria(pred, label.to(torch.uint8).cuda())
            train_loss += loss.item()
            error = torch.sum(pred.argmax(1) != label.flatten().cuda())


        # print(loss.item())
        if args.method == 'cnnvote':
            error = error.cpu().numpy() / 64
        elif args.method == 'attmil':
            error, _ = model.calculate_classification_error(data, bag_label)
        else:
            error, _ = model.calculate_classification_error(vgg16(data[0]).unsqueeze(0), bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    logger.info('Epoch:  %d, Loss: %.3f, Train accuracy: %.2f' % (epoch, train_loss, 100 - train_error*100))


best_error = 1.

def val(epoch):
    model.eval()
    val_loss = 0.
    val_error = 0.
    # global loss

    for batch_idx, (data, label) in enumerate(val_loader):
        bag_label = label[0]
        if bag_label == 1:
            instance_labels = torch.Tensor([True for i in range(200)])
        else:
            instance_labels = torch.Tensor([False for i in range(200)])
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        
        if args.method == 'attmil':
            loss, _ = model.calculate_objective(data, bag_label)
            val_loss += loss.data[0].cpu().numpy()[0]
        elif args.method == 'sett':
            pred = model(vgg16(data[0]).unsqueeze(0)).unsqueeze(0)
            loss = criteria(pred, bag_label.to(torch.uint8))
            val_loss += loss.item()
        elif  args.method == 'deepset':
            pred = model(vgg16(data[0]).unsqueeze(0))
            loss = criteria(pred[0], bag_label.to(torch.uint8))
            val_loss += loss.item()
        elif  args.method == 'cnnvote':
            pred = model(data[:, 0])
            loss = criteria(pred, label.to(torch.uint8).cuda())
            val_loss += loss.item()
            error = torch.sum(pred.argmax(1) != label.flatten().cuda())


        # print(loss.item())
        if args.method == 'cnnvote':
            error = error.cpu().numpy() / 64
        elif args.method == 'attmil':
            error, _ = model.calculate_classification_error(data, bag_label)
        else:
            error, _ = model.calculate_classification_error(vgg16(data[0]).unsqueeze(0), bag_label)
        val_error += error

    val_error /= len(val_loader)
    val_loss /= len(val_loader)

    # print(val_error, best_error)
    if val_error < best_error:
        torch.save(model.state_dict(), args.exp_path + '/' + args.Emocat + '.pth')
        print('ckpt updated')
        global temperr 
        temperr = val_error

    logger.info('Epoch:  %d, Val Loss: %.3f, Val accuracy: %.2f' % (epoch, val_loss, 100 - val_error*100))



def val_MV(epoch):
    model.eval()
    val_loss = 0.
    val_error = 0.
    # global loss

    for batch_idx, (data, label) in enumerate(val_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        pred = model(data[0].cuda())
        res = pred.argmax(1).mode(0).values

        
        error = int(res.item() != bag_label.item())        
        val_error += error

    val_error /= len(val_loader)
    val_loss /= len(val_loader)

    # print(val_error, best_error)
    if val_error < best_error:
        torch.save(model.state_dict(), args.exp_path + '/' + args.Emocat + '.pth')
        print('ckpt updated')
        global temperr 
        temperr = val_error

    logger.info('Epoch:  %d, Val Loss: %.3f, Val accuracy: %.2f' % (epoch, val_loss, 100 - val_error*100))


def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # calculate loss and metrics
        if args.method == 'attmil':
            loss, _ = model.calculate_objective(data, bag_label)
            test_loss += loss.data[0].cpu().numpy()[0]
        elif args.method == 'sett':
            pred = model(vgg16(data[0]).unsqueeze(0)).unsqueeze(0)
            loss = criteria(pred, bag_label.to(torch.uint8))
            test_loss += loss.item()
        elif args.method == 'deepset':
            pred = model(vgg16(data[0]).unsqueeze(0))
            loss = criteria(pred[0], bag_label.to(torch.uint8))
            test_loss += loss.item()
        elif  args.method == 'cnnvote':
            pred = model(data[:, 0])
            loss = criteria(pred, label.to(torch.uint8).cuda())
            test_loss += loss.item()
            error = torch.sum(pred.argmax(1) != label.flatten().cuda())


        # print(loss.item())
        if args.method == 'cnnvote':
            error = error.cpu().numpy() / 64
        elif args.method == 'attmil':
            error, _ = model.calculate_classification_error(data, bag_label)
        else:
            error, _ = model.calculate_classification_error(vgg16(data[0]).unsqueeze(0), bag_label)
        # print(loss.item())
        
        test_error += error

    # calculate loss and error for epoch
    test_loss /= len(test_loader)
    test_error /= len(test_loader)


    logger.info('Test Loss: %.3f, Test accuracy: %.2f' % (test_loss, 100 - test_error*100))



def test_MV():
    model.eval()
    test_loss = 0.
    test_error = 0.
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        pred = model(data[0].cuda())
        res = pred.argmax(1).mode(0).values

        
        error = int(res.item() != bag_label.item())        

        test_error += error

    # calculate loss and error for epoch
    test_loss /= len(test_loader)
    test_error /= len(test_loader)


    logger.info('Test Loss: %.3f, Test accuracy: %.2f' % (test_loss, 100 - test_error*100))


if __name__ == "__main__":
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if args.method != 'cnnvote':
            val(epoch)
            test()
        else:
            # val_MV(epoch)
            val(epoch)
            test_MV()
        best_error = temperr

    print('Start Testing')
    model.load_state_dict(torch.load(args.exp_path + '/' + args.Emocat + '.pth'))
    model.eval()
    if args.method != 'cnnvote':
        test()
    else:
        test_MV()
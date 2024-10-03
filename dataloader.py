"""Pytorch Dataset object that loads perfectly balanced MNIST dataset in bag form."""

from cv2 import split
from matplotlib.transforms import Transform
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import random

img_to_tensor = transforms.Compose([
                            transforms.Resize((32,32)), 
                            transforms.ToTensor(),])

img_to_tensor_224 = transforms.Compose([
                            transforms.CenterCrop((224, 224)), 
                            transforms.ToTensor(),])
img_to_tensor_224_imn  = transforms.Compose([
                            transforms.Resize((256, 256)), 
                            transforms.CenterCrop((224, 224)), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            transforms.ToTensor(),])

class E6instance(data_utils.Dataset):
    def __init__(self, train, cat, data_path='./data/emotion_mixed_10/', num_users=None):

        self.dir = data_path
        self.cat = cat
        self.pos_all = []
        self.neg_all = []
        self.train = train
        # self.label_dir = label_dir
            
        #  check this function to make sure that the label_ls is correct.
        self.label_ls = []
        self.num_users = num_users

        # for i in range(len(self.get_users(self.train)[1])):
        #     temp = self.get_label(i)
        #     for item in self.get_bags(i):
        #         print(item)
        #         self.label_ls.append([temp])
        
        
        for item in os.listdir(self.dir + self.cat):
            self.pos_all.append(self.dir + self.cat + '/' + item + '/')

        for sub_dir in os.listdir(self.dir):
            if sub_dir != self.cat:
                for item in os.listdir(self.dir + sub_dir):
                    self.neg_all.append(self.dir + sub_dir + '/' + item + '/')

        random.seed(10)
        random.shuffle(self.neg_all)

        self.imgs = []
        for i in tqdm(range(len(self.get_users(self.train)[0]))):
            # for item in self.get_bags(i):
                # self.imgs.append(item)
            self.imgs.append(self.get_bags(i))

    def get_users(self, train):
        cat_all = ['anger', 'disgust', 'fear', 'joy', 'sadness']


        if self.num_users is None:
            self.pos_all = self.pos_all
            self.neg_all = self.neg_all[:int(len(self.pos_all))]
        else:
            self.pos_all = self.pos_all[:self.num_users]
            self.neg_all = self.neg_all[:int(len(self.pos_all))][:self.num_users]   

        # NEED TO BE ADJUSTED FOR DIFFERENT LENGTH!!
        ptrain = self.pos_all[:int(0.8 * len(self.pos_all))]
        pval = self.pos_all[int(0.8 * len(self.pos_all)):int(0.9 * len(self.pos_all))]
        ptest = self.pos_all[int(0.9 * len(self.pos_all)):] 
        ntrain = self.neg_all[:int(0.8 * len(self.neg_all))]
        nval = self.neg_all[int(0.8 * len(self.neg_all)):int(0.9 * len(self.neg_all))]
        ntest = self.neg_all[int(0.9 * len(self.neg_all)):]
        label_train = np.append(np.ones(np.array(ptrain).shape).flatten(), np.zeros(np.array(ntrain).shape).flatten())
        label_val = np.append(np.ones(np.array(pval).shape).flatten(), np.zeros(np.array(nval).shape).flatten())
        label_test = np.append(np.ones(np.array(ptest).shape).flatten(), np.zeros(np.array(ntest).shape).flatten())

        if train == 'train':
            return np.append(ptrain, ntrain), label_train
        elif train == 'val':
            return np.append(pval, nval), label_val
        else:
            return np.append(ptest, ntest), label_test

    def get_bags(self, i):
        img_ls = torch.empty((0, 3, 32, 32))
        temp = self.get_users(self.train)[0][i]
        for item in os.listdir(temp):
            img = Image.open(temp + '/' + item.decode("utf-8")).convert('RGB')
            img = img_to_tensor(img).unsqueeze(0)
            img_ls = torch.cat((img_ls, img), dim = 0)
        return img_ls


    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        for item in self.get_users(self.train)[1]:
            self.label_ls.append([item])
        bag = self.imgs[index]
        label = self.label_ls[index]
        return bag, label
    

class E6instance224(data_utils.Dataset):
    def __init__(self, train, cat, data_path='./data/emotion_mixed_10/', num_users=None):

        self.dir = data_path
        self.cat = cat
        self.pos_all = []
        self.neg_all = []
        self.train = train
        # self.label_dir = label_dir
            
        #  check this function to make sure that the label_ls is correct.
        self.label_ls = []
        self.num_users = num_users

        # for i in range(len(self.get_users(self.train)[1])):
        #     temp = self.get_label(i)
        #     for item in self.get_bags(i):
        #         print(item)
        #         self.label_ls.append([temp])
        
        
        for item in os.listdir(self.dir + self.cat):
            self.pos_all.append(self.dir + self.cat + '/' + item + '/')

        for sub_dir in os.listdir(self.dir):
            if sub_dir != self.cat:
                for item in os.listdir(self.dir + sub_dir):
                    self.neg_all.append(self.dir + sub_dir + '/' + item + '/')

        random.seed(10)
        random.shuffle(self.neg_all)

        if self.num_users is None:
            self.pos_all = self.pos_all
            self.neg_all = self.neg_all[:int(len(self.pos_all))]
        else:
            self.pos_all = self.pos_all[:self.num_users]
            self.neg_all = self.neg_all[:int(len(self.pos_all))][:self.num_users]   
            
        self.imgs = []
        for i in tqdm(range(len(self.get_users(self.train)[0]))):
            # for item in self.get_bags(i):
                # self.imgs.append(item)
            self.imgs.append(self.get_bags(i))

    def get_users(self, train):
        cat_all = ['anger', 'disgust', 'fear', 'joy', 'sadness']



        # NEED TO BE ADJUSTED FOR DIFFERENT LENGTH!!
        ptrain = self.pos_all[:int(0.8 * len(self.pos_all))]
        pval = self.pos_all[int(0.8 * len(self.pos_all)):int(0.9 * len(self.pos_all))]
        ptest = self.pos_all[int(0.9 * len(self.pos_all)):] 
        ntrain = self.neg_all[:int(0.8 * len(self.neg_all))]
        nval = self.neg_all[int(0.8 * len(self.neg_all)):int(0.9 * len(self.neg_all))]
        ntest = self.neg_all[int(0.9 * len(self.neg_all)):]
        
        # ptrain = ptrain[:int(0.5*len(ptrain))]
        # pval = pval[:int(0.5*len(pval))]
        # ptest = ptest[:int(0.5*len(ptest))]
        # ntrain = ntrain[:int(0.5*len(ntrain))]
        # nval = nval[:int(0.5*len(nval))]
        # ntest = ntest[:int(0.5*len(ntest))]
        
        label_train = np.append(np.ones(np.array(ptrain).shape).flatten(), np.zeros(np.array(ntrain).shape).flatten())
        label_val = np.append(np.ones(np.array(pval).shape).flatten(), np.zeros(np.array(nval).shape).flatten())
        label_test = np.append(np.ones(np.array(ptest).shape).flatten(), np.zeros(np.array(ntest).shape).flatten())



        if train == 'train':
            return np.append(ptrain, ntrain), label_train
        elif train == 'val':
            return np.append(pval, nval), label_val
        else:
            return np.append(ptest, ntest), label_test

    def get_bags(self, i):
        img_ls = torch.empty((0, 3, 224, 224))
        temp = self.get_users(self.train)[0][i]
        for item in os.listdir(temp):
            img = Image.open(temp + '/' + item.decode("utf-8")).convert('RGB')
            img = img_to_tensor_224(img).unsqueeze(0)
            img_ls = torch.cat((img_ls, img), dim = 0)
        return img_ls


    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        for item in self.get_users(self.train)[1]:
            self.label_ls.append([item])
        bag = self.imgs[index]
        label = self.label_ls[index]
        return bag, label



class E6instance224_imntrans(data_utils.Dataset):
    def __init__(self, train, cat, data_path='./data/emotion_mixed_10/', num_users=None):

        self.dir = data_path
        self.cat = cat
        self.pos_all = []
        self.neg_all = []
        self.train = train
        # self.label_dir = label_dir
            
        #  check this function to make sure that the label_ls is correct.
        self.label_ls = []
        self.num_users = num_users

        # for i in range(len(self.get_users(self.train)[1])):
        #     temp = self.get_label(i)
        #     for item in self.get_bags(i):
        #         print(item)
        #         self.label_ls.append([temp])
        
        
        for item in os.listdir(self.dir + self.cat):
            self.pos_all.append(self.dir + self.cat + '/' + item + '/')

        for sub_dir in os.listdir(self.dir):
            if sub_dir != self.cat:
                for item in os.listdir(self.dir + sub_dir):
                    self.neg_all.append(self.dir + sub_dir + '/' + item + '/')

        random.seed(10)
        random.shuffle(self.neg_all)

        if self.num_users is None:
            self.pos_all = self.pos_all
            self.neg_all = self.neg_all[:int(len(self.pos_all))]
        else:
            self.pos_all = self.pos_all[:self.num_users]
            self.neg_all = self.neg_all[:int(len(self.pos_all))][:self.num_users]   
            
        self.imgs = []
        for i in tqdm(range(len(self.get_users(self.train)[0]))):
            # for item in self.get_bags(i):
                # self.imgs.append(item)
            self.imgs.append(self.get_bags(i))

    def get_users(self, train):
        cat_all = ['anger', 'disgust', 'fear', 'joy', 'sadness']



        # NEED TO BE ADJUSTED FOR DIFFERENT LENGTH!!
        ptrain = self.pos_all[:int(0.8 * len(self.pos_all))]
        pval = self.pos_all[int(0.8 * len(self.pos_all)):int(0.9 * len(self.pos_all))]
        ptest = self.pos_all[int(0.9 * len(self.pos_all)):] 
        ntrain = self.neg_all[:int(0.8 * len(self.neg_all))]
        nval = self.neg_all[int(0.8 * len(self.neg_all)):int(0.9 * len(self.neg_all))]
        ntest = self.neg_all[int(0.9 * len(self.neg_all)):]
        
        # ptrain = ptrain[:int(0.5*len(ptrain))]
        # pval = pval[:int(0.5*len(pval))]
        # ptest = ptest[:int(0.5*len(ptest))]
        # ntrain = ntrain[:int(0.5*len(ntrain))]
        # nval = nval[:int(0.5*len(nval))]
        # ntest = ntest[:int(0.5*len(ntest))]
        
        label_train = np.append(np.ones(np.array(ptrain).shape).flatten(), np.zeros(np.array(ntrain).shape).flatten())
        label_val = np.append(np.ones(np.array(pval).shape).flatten(), np.zeros(np.array(nval).shape).flatten())
        label_test = np.append(np.ones(np.array(ptest).shape).flatten(), np.zeros(np.array(ntest).shape).flatten())



        if train == 'train':
            return np.append(ptrain, ntrain), label_train
        elif train == 'val':
            return np.append(pval, nval), label_val
        else:
            return np.append(ptest, ntest), label_test

    def get_bags(self, i):
        img_ls = torch.empty((0, 3, 224, 224))
        temp = self.get_users(self.train)[0][i]
        for item in os.listdir(temp):
            img = Image.open(temp + '/' + item.decode("utf-8")).convert('RGB')
            img = img_to_tensor_224_imn(img).unsqueeze(0)
            img_ls = torch.cat((img_ls, img), dim = 0)
        return img_ls


    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        for item in self.get_users(self.train)[1]:
            self.label_ls.append([item])
        bag = self.imgs[index]
        label = self.label_ls[index]
        return bag, label

class E6feature(data_utils.Dataset):
    def __init__(self, train, cat, data_path='./data/emotion_mixed_10_features/', num_users=None):

        self.dir = data_path
        self.cat = cat
        self.pos_all = []
        self.neg_all = []
        self.train = train
        # self.label_dir = label_dir
            
        #  check this function to make sure that the label_ls is correct.
        self.label_ls = []
        self.num_users = num_users

        # for i in range(len(self.get_users(self.train)[1])):
        #     temp = self.get_label(i)
        #     for item in self.get_bags(i):
        #         print(item)
        #         self.label_ls.append([temp])
        
        
        for item in os.listdir(self.dir + self.cat):
            self.pos_all.append(self.dir + self.cat + '/' + item + '/')

        for sub_dir in os.listdir(self.dir):
            if sub_dir != self.cat:
                for item in os.listdir(self.dir + sub_dir):
                    self.neg_all.append(self.dir + sub_dir + '/' + item + '/')

        random.seed(10)
        random.shuffle(self.neg_all)

        if self.num_users is None:
            self.pos_all = self.pos_all
            self.neg_all = self.neg_all[:int(len(self.pos_all))]
        else:
            self.pos_all = self.pos_all[:self.num_users]
            self.neg_all = self.neg_all[:int(len(self.pos_all))][:self.num_users]   

        self.imgs = []
        for i in tqdm(range(len(self.get_users(self.train)[0]))):
            # for item in self.get_bags(i):
                # self.imgs.append(item)
            self.imgs.append(self.get_bags(i))

    def get_users(self, train):
        cat_all = ['anger', 'disgust', 'fear', 'joy', 'sadness']




        # NEED TO BE ADJUSTED FOR DIFFERENT LENGTH!!
        ptrain = self.pos_all[:int(0.8 * len(self.pos_all))]
        pval = self.pos_all[int(0.8 * len(self.pos_all)):int(0.9 * len(self.pos_all))]
        ptest = self.pos_all[int(0.9 * len(self.pos_all)):] 
        ntrain = self.neg_all[:int(0.8 * len(self.neg_all))]
        nval = self.neg_all[int(0.8 * len(self.neg_all)):int(0.9 * len(self.neg_all))]
        ntest = self.neg_all[int(0.9 * len(self.neg_all)):]
        label_train = np.append(np.ones(np.array(ptrain).shape).flatten(), np.zeros(np.array(ntrain).shape).flatten())
        label_val = np.append(np.ones(np.array(pval).shape).flatten(), np.zeros(np.array(nval).shape).flatten())
        label_test = np.append(np.ones(np.array(ptest).shape).flatten(), np.zeros(np.array(ntest).shape).flatten())

        if train == 'train':
            return np.append(ptrain, ntrain), label_train
        elif train == 'val':
            return np.append(pval, nval), label_val
        else:
            return np.append(ptest, ntest), label_test

    def get_bags(self, i):
        feature_ls = torch.empty((0, 4096))
        temp = self.get_users(self.train)[0][i]
        for item in os.listdir(temp):
            feature = torch.Tensor(np.load(temp + '/' + item.decode("utf-8")))
            # feature = img_to_tensor(feature).unsqueeze(0)
            feature_ls = torch.cat((feature_ls, feature), dim = 0)
        return feature_ls


    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        for item in self.get_users(self.train)[1]:
            self.label_ls.append([item])
        bag = self.imgs[index]
        label = self.label_ls[index]
        return bag, label
    




img_to_tensor_224 = transforms.Compose([
                            # transforms.Resize((224,224)), 
                            transforms.CenterCrop(224),

                            # transforms.Resize((32, 32)), 

                            transforms.ToTensor(),])


class E6singleimage(data_utils.Dataset):
    def __init__(self, train, cat, data_path='./data/emotion_mixed_10/', num_users=None):

        self.dir = data_path
        self.cat = cat
        self.pos_all = []
        self.neg_all = []
        self.train = train
        # self.label_dir = label_dir
        self.pos_all_img = []
        self.neg_all_img = []
        self.num_users = num_users

        
        for item in os.listdir(self.dir + self.cat):
            self.pos_all.append(self.dir + self.cat + '/' + item + '/')

        for sub_dir in os.listdir(self.dir):
            if sub_dir != self.cat:
                for item in os.listdir(self.dir + sub_dir):
                    self.neg_all.append(self.dir + sub_dir + '/' + item + '/')
        
        random.seed(10)
        random.shuffle(self.neg_all)

        if self.num_users is None:
            self.pos_all = self.pos_all
            self.neg_all = self.neg_all[:int(len(self.pos_all))]
        else:
            self.pos_all = self.pos_all[:self.num_users]
            self.neg_all = self.neg_all[:int(len(self.pos_all))][:self.num_users]  

        for item in self.pos_all:
            for img in os.listdir(item):
                if img.split('.')[-1] == 'jpg':
                    self.pos_all_img.append(item + '/' + img)

        for item in self.neg_all:
             for img in os.listdir(item):
                if img.split('.')[-1] == 'jpg':
                    self.neg_all_img.append(item + '/' + img)

        #  check this function to make sure that the label_ls is correct.
        self.label_ls = []
        self.num_users = num_users
        self.get_users_res_img = self.get_users(self.train)[0]
        self.get_users_res_label = self.get_users(self.train)[1]
        # for i in range(len(self.get_users(self.train)[1])):
        #     temp = self.get_label(i)
        #     for item in self.get_bags(i):
        #         print(item)
        #         self.label_ls.append([temp])
        



    def get_users(self, train):
        cat_all = ['anger', 'disgust', 'fear', 'joy', 'sadness']

        # NEED TO BE ADJUSTED FOR DIFFERENT LENGTH!!
        ptrain = self.pos_all_img[:int(0.8 * len(self.pos_all_img))]
        pval = self.pos_all_img[int(0.8 * len(self.pos_all_img)):int(0.9 * len(self.pos_all_img))]
        ptest = self.pos_all_img[int(0.9 * len(self.pos_all_img)):] 
        ntrain = self.neg_all_img[:int(0.8 * len(self.neg_all_img))]
        nval = self.neg_all_img[int(0.8 * len(self.neg_all_img)):int(0.9 * len(self.neg_all_img))]
        ntest = self.neg_all_img[int(0.9 * len(self.neg_all_img)):]
        label_train = np.append(np.ones(np.array(ptrain).shape).flatten(), np.zeros(np.array(ntrain).shape).flatten())
        label_val = np.append(np.ones(np.array(pval).shape).flatten(), np.zeros(np.array(nval).shape).flatten())
        label_test = np.append(np.ones(np.array(ptest).shape).flatten(), np.zeros(np.array(ntest).shape).flatten())

        if train == 'train':
            return np.append(ptrain, ntrain), label_train
        elif train == 'val':
            return np.append(pval, nval), label_val
        else:
            return np.append(ptest, ntest), label_test

    def get_img(self, i):
        temp = self.get_users_res_img[i]
        label = self.get_users_res_label[i]
        img = Image.open(temp).convert('RGB')
        img = img_to_tensor_224(img).unsqueeze(0)
        return img, label


    def __len__(self):
        return len(self.get_users_res_img)
        
    def __getitem__(self, index):
        bag = self.get_img(index)[0]
        label = self.get_img(index)[1]
        return bag, label
    


if __name__ == "__main__":
#     print(next(iter(PFinstance('train', 'O'))))
    a = E6feature(train='test', cat ='anger')
    print(next(iter(a))[0].shape)
    print(len(a))

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
from tqdm import tqdm

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils import data_transforms, ddl_direct, sample_smaller_test, Threshold_dict, load_from_list, Normal_Threshold_dict
from group_model import MV, weighted, model_select_MV, model_select_fusion

plt.ion()
device = 'cuda:0'



users_list = sorted(os.listdir('./data/PF_all/'))
img_root = './data/PF_all/'




for personality_trait in tqdm(['A']):
    for net in ['vgg']:
        for user_type in ['strong']:
            
            # for repeat in range(10):
            save_dir = './experimental_results/' + personality_trait + '_' + net + '_' + user_type + '/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            t_dict = Threshold_dict[personality_trait][net + '_' + user_type + '_' + personality_trait]
            t_w = Normal_Threshold_dict[personality_trait][net + '_' + user_type + '_' + personality_trait]

            if user_type == 'weak':
                data_loaded = np.load('./data/weak' + personality_trait + '.npy', allow_pickle=True)
            else:
                data_loaded = np.load('./data/' + personality_trait + '.npy', allow_pickle=True)

            num_im_profile = 20
            t_test = sample_smaller_test(img_root, data_loaded.item()['test_high'], data_loaded.item()['test_low'], normal_sample_size = 20, adv_size = 5, user_num = 20)

            normal_added = np.empty((0, 20))
            for item in range(len(t_test)):
                all_ls = os.listdir(img_root + users_list[t_test[item][0]])
                np.random.seed(1234)
                left = np.array(all_ls)[~np.isin(all_ls, t_test[item][2])]
                normal_added = np.append(normal_added, np.array([left[j] for j in np.random.randint(180, size=20)]).reshape((1, -1)), axis = 0)

            
            # normal_added = np.empty((0, 20))
            # for item in range(len(t_test)):
            #     all_ls = os.listdir(img_root + users_list[t_test[item][0]])
            #     np.random.seed(repeat)
            #     left = np.array(all_ls)[~np.isin(all_ls, t_test[item][2])]
            #     normal_added = np.append(normal_added, np.array([left[j] for j in np.random.randint(180, size=20)]).reshape((1, -1)), axis = 0)

            # # majority voting


            model_ft = model_select_MV(net, personality_trait)

            mv_normal = {}
            mv_normal_roc = {}
            roc_input =[]
            acc = []
            for u_idx, item in (enumerate(t_test[:, 0])):
                temp_list = []

                temp_ls = [img_root + users_list[item] + '/'+ j for j in t_test[u_idx][2]] + [img_root + users_list[item] + '/'+ j for j in normal_added[u_idx][:2]]
                label, roc = MV(temp_ls, t_test, u_idx, model_ft)
                acc.append(label)
                roc_input.append(roc)

            
            mv_normal = np.array(np.mean(acc)).reshape((1, -1))
            
            mv_normal_roc = np.array(roc_input)
            print(mv_normal)
            

            np.save(save_dir + '20_mv_normal.npy', mv_normal)
            np.save(save_dir + '20_mv_normal_roc.npy', mv_normal_roc)   


            mv_normal = {}
            mv_normal_roc = {}
            for case in ['6', '10', '20']:
            # for case in ['1']:

                # t_test = sample_smaller_test(img_root, data_loaded.item()['test_high'], data_loaded.item()['test_low'], normal_sample_size = num_im_profile, adv_size = int(case), user_num = 20)
                roc_input =[]
                acc = []
                for u_idx, item in (enumerate(t_test[:, 0])):
                    temp_list = []
                    # print(users_list[item])
                    temp_ls = [img_root + users_list[item] + '/'+ j for j in t_test[u_idx][2]] + [img_root + users_list[item] + '/' + j for j in normal_added[u_idx][:int(case)]]
                    label, roc = MV(temp_ls, t_test, u_idx, model_ft)
                    # print(label)
                    acc.append(label)
                    roc_input.append(roc)


                if case not in mv_normal.keys():
                    mv_normal[case] = np.array(np.mean(acc)).reshape((1, -1))
                else:
                    mv_normal[case] = np.append(mv_normal[case], np.mean(np.array(acc)).reshape((1, -1)), axis = 0)


                if case not in mv_normal_roc.keys():
                    mv_normal_roc[case] = np.array(roc_input)
                else:
                    mv_normal_roc[case] = np.append(mv_normal_roc[case], np.array(roc_input), axis = 1)

            np.save(save_dir + 'mv_normal.npy', mv_normal)
            np.save(save_dir + 'mv_normal_roc.npy', mv_normal_roc)



            # weight normal

            model_ft, final_classifier = model_select_fusion(net, personality_trait)

            w_normal = {}
            w_normal_roc = {}

            roc_input =[]
            acc = []
            for u_idx, item in (enumerate(t_test[:, 0])):
                temp_ls = [img_root + users_list[item] + '/'+ j for j in t_test[u_idx][2]] 
                label, roc = weighted(temp_ls, t_test, u_idx, model_ft, final_classifier, personality_trait, threshold = t_w, net=net, average = False)

                acc.append(label)
                roc_input.append(roc)

        
            w_normal= np.array(np.mean(acc)).reshape((1, -1))
    

            w_normal_roc = np.array(roc_input)
            

            np.save(save_dir + '20_w_normal.npy', w_normal)
            np.save(save_dir + '20_w_normal_roc.npy', w_normal_roc)


            w_normal = {}
            w_normal_roc = {}

            for case in ['6', '10', '20']:
                # t_test = sample_smaller_test(img_root, data_loaded.item()['test_high'], data_loaded.item()['test_low'], normal_sample_size = num_im_profile, adv_size = int(case), user_num = 20)
                roc_input =[]
                acc = []
                for u_idx, item in (enumerate(t_test[:, 0])):
                    temp_ls = [img_root + users_list[item] + '/'+ j for j in t_test[u_idx][2]] + [img_root + users_list[item] + '/'+ j for j in normal_added[u_idx][:int(case)]]
                    label, roc = weighted(temp_ls, t_test, u_idx, model_ft, final_classifier, personality_trait, threshold = t_dict[int(case) + 5], net=net, average = False)

                    acc.append(label)
                    roc_input.append(roc)

                if case not in w_normal.keys():
                    w_normal[case] = np.array(np.mean(acc)).reshape((1, -1))
                else:
                    w_normal[case] = np.append(w_normal[case], np.mean(np.array(acc)).reshape((1, -1)), axis = 0)

                if case not in w_normal_roc.keys():
                    w_normal_roc[case] = np.array(roc_input)
                else:
                    w_normal_roc[case] = np.append(w_normal_roc[case], np.array(roc_input), axis = 1)

            np.save(save_dir + 'w_normal.npy', w_normal)
            np.save(save_dir + 'w_normal_roc.npy', w_normal_roc)
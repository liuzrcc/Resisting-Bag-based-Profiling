from __future__ import print_function
import sys
sys.path.append('./')
from random import shuffle

import numpy as np
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
from dataloader import *
from model import SetTransformer, DeepSet, ResNet18
import os
import torch.nn as nn



from attack import *



parser = argparse.ArgumentParser(description='Blackbox Pivoting Addtions')
parser.add_argument('--model', type=str, default='sett', help='target model')
parser.add_argument('--Emocat', type=str, default='disgust', help='Emotion Profile dataset')
parser.add_argument('--IMN_val_dir', type=str, default='/sata1/ILSVRC2012/val/', help='Emotion Profile dataset')

args = parser.parse_args()

target_method = args.model
emocat = args.Emocat
NUM_ADD = 10


loader_kwargs = {'num_workers': 8, 'pin_memory': True}

 
test_loader= data_utils.DataLoader(E6instance224('test', emocat),
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs)

model = models.resnet50(weights="IMAGENET1K_V2")
modules=list(model.children())[:-1]
model=nn.Sequential(*modules)
for p in model.parameters():
    p.requires_grad = False
model.cuda()
pass



data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transform32 = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])

vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
vgg16.classifier = vgg16.classifier[:-1]
vgg16.cuda()
vgg16.eval()
pass

candidate_ls = [os.path.join(root, name)
            for root, dirs, files in os.walk(args.IMN_val_dir)
            for name in files
            if name.endswith((".JPEG"))]

shuffle(candidate_ls)

print("############Preparing background data############")

bk_data = torch.rand(50000, 2048)
for idx, img in enumerate(candidate_ls):
        temp = Image.open(img).convert('RGB')
        temp = data_transform(temp)
        feat = model(temp.cuda().unsqueeze(0))
        bk_data[idx] = feat.view(1, 2048)
        
bk_data.cuda()

def return_NIADD_black(profile_feature, bk_data, NUM_return=10):
    candidates_bk = torch.cdist(profile_feature, bk_data).mean(dim=0)
    top10_idx = sorted(range(len(candidates_bk)), key=lambda i: candidates_bk[i], reverse=False)[:NUM_return]
    return [candidate_ls[item] for item in top10_idx]



print("############Attack" + target_method + "on Emotion Profile" + emocat + "############")
loader_kwargs = {'num_workers': 8, 'pin_memory': True}

test_loader= data_utils.DataLoader(E6instance224('test', emocat),
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs)


if target_method == 'deepset':
    model_tar = DeepSet(dim_input=4096,num_outputs=1, dim_output=2)
elif target_method == 'sett':
    model_tar = SetTransformer()

model_tar.load_state_dict(torch.load('./exp/e2e-emotion-' + target_method + '-' + emocat + '/'  + emocat + '.pth'))
model_tar.cuda()
model_tar.eval()

            

imgs_ls = [os.path.join(root, name)
            for root, dirs, files in os.walk(args.IMN_val_dir)
            for name in files
            if name.endswith((".JPEG"))]

random.Random(1234).shuffle(imgs_ls)

candidate_ls = imgs_ls

list_of_NUM_ADD = random.Random(1234).sample(imgs_ls, NUM_ADD)

ct = 0
for batch_idx, (data, label) in enumerate(test_loader):
    pred = model_tar(vgg16(data.cuda()[0]).unsqueeze(0))
    if target_method == 'deepset':
        if label[0].item() == pred[0][0].argmax(0).item():
            ct +=1 
    elif target_method == 'sett':
        if pred.argmax().item() == label[0].item():
            ct +=1 
print("#############Original accuacy (before pivoting) is", ct / test_loader.__len__())


perturbations = torch.empty(0, 10, 3, 224, 224)

for item in test_loader:
    delta = pgd_linf_pire(model, item[0][0].cuda(), epsilon=16/255., alpha=2/255., num_iter=100)
    perturbations = torch.cat((perturbations, delta.cpu().unsqueeze(0)), dim=0)

# torch.save(perturbations, emocat+ "noiseADD.pt")

ct = 0
for batch_idx, (data, label) in enumerate(test_loader):
    pred = model_tar(vgg16(torch.cat((data.cuda(), perturbations[batch_idx].unsqueeze(0).cuda()), dim=1)[0]).unsqueeze(0))
    if target_method == 'deepset':
        if label[0].item() == pred[0][0].argmax(0).item():
            ct +=1 
    elif target_method == 'sett':
        if pred.argmax().item() == label[0].item():
            ct +=1 
print("#############AdvN accuacy is", ct / test_loader.__len__())



candidate_imgs_tensor = torch.zeros(1, 10, 3, 224, 224)

for idx, item in enumerate(list_of_NUM_ADD):
    img = Image.open(item).convert('RGB')
    candidate_imgs_tensor[:, idx] = data_transform(img).cuda().unsqueeze(0)
    
    
perturbations = torch.empty(0, 10, 3, 224, 224)

for item in test_loader:
    delta = pgd_linf_pire_advimage(model, item[0][0].cuda(), candidate_imgs_tensor.cuda()[0], epsilon=16/255., alpha=2/255., num_iter=100)
    perturbations = torch.cat((perturbations, delta.cpu().unsqueeze(0)), dim=0)

ct = 0
for batch_idx, (data, label) in enumerate(test_loader):
    pred = model_tar(vgg16(torch.cat((data.cuda(), torch.clamp(perturbations[batch_idx].unsqueeze(0).cuda() + candidate_imgs_tensor.cuda(), 0, 1)), dim=1)[0]).unsqueeze(0))
    if target_method == 'deepset':
        if label[0].item() == pred[0][0].argmax(0).item():
            ct += 1
    elif target_method == 'sett':
        if pred.argmax().item() == label[0].item():
            ct += 1 
print("#############AdvPI accuracy is", ct / test_loader.__len__())

ct = 0

for batch_idx, (data, label) in enumerate(test_loader):
    bag_label = label[0].bool()

    data, bag_label = data.cuda(), bag_label.cuda()
    data, bag_label = Variable(data), Variable(bag_label)

    NIADD_tensor = return_NIADD_black(model(data[0]).reshape(-1, 2048), bk_data.cuda())

    niadd_neg = torch.zeros(1, NUM_ADD, 3, 224, 224)

    for idx,item in enumerate(NIADD_tensor):
        niadd_neg[:, idx] = data_transform(Image.open(item).convert('RGB')).unsqueeze(0).cuda()


    pred = model_tar(vgg16(torch.cat((data, niadd_neg.cuda()), dim=1)[0]).unsqueeze(0))
    # pred = model(data)

    
    if target_method == 'deepset':
        if label[0].item() == pred[0][0].argmax(0).item():
            ct += 1
    elif target_method == 'sett':
        if pred.cpu().argmax() == label[0]:
            ct += 1

print("############# NatI accuracy is", ct / test_loader.__len__())

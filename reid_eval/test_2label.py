
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function, division

import sys
sys.path.append('..')
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
from reIDmodel import ft_net, ft_netAB, ft_net_dense, PCB, PCB_test
import glob
import gradcam
import guided_backprop
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default=90000, type=int, help='80000')
parser.add_argument('--test_dir',default='../../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='test', type=str, help='save model path')
parser.add_argument('--config', type=str)
parser.add_argument('--batchsize', default=80, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--att_maps', action='store_true', help='to save att maps')
parser.add_argument('--gradcam', action='store_true', help='to compute gradcam')
parser.add_argument('--exp_id', type=str)
parser.add_argument('--model_id', type=str)
parser.add_argument('--which_trial',type=int)
parser.add_argument('--batch_inds', type=int,nargs="*",help="The batch index to save the maps of.")

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_nonorm = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor()])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


data_dir = test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=0) for x in ['gallery','query','multi-query']}

image_dataset_nonorm = datasets.ImageFolder(os.path.join(data_dir,'gallery') ,data_transforms_nonorm)
dataloader_nonorm = torch.utils.data.DataLoader(image_dataset_nonorm, batch_size=opt.batchsize,shuffle=False, num_workers=0)

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def addOrRemoveModule(net,weights):

    exKeyWei = list(weights.keys())[0]
    exKeyNet = list(net.state_dict().keys())[0]

    print(exKeyWei,exKeyNet)

    if exKeyWei.find("module") != -1 and exKeyNet.find("module") == -1:
        #remove module
        newWeights = {}
        for param in weights:
            newWeights[param.replace("module.","")] = weights[param]
        weights = newWeights

    if exKeyWei.find("module") == -1 and exKeyNet.find("module") != -1:
        #add module
        newWeights = {}
        for param in weights:
            newWeights["module."+param] = weights[param]

        weights = newWeights

    return weights

def load_network(network,save_path=None):
    if save_path is None:
        save_path = os.path.join('../outputs',name,'checkpoints/id_%08d.pt'%opt.which_epoch)
    state_dict = torch.load(save_path)

    state_dict["a"] = addOrRemoveModule(network,state_dict["a"])

    if list(network.state_dict().keys())[0].find("module.") != -1:
        classKeys = ["module.classifier1.add_block.0.weight","module.classifier2.add_block.0.weight"]
    else:
        classKeys = ["classifier1.add_block.0.weight","classifier2.add_block.0.weight"]

    for classKey in classKeys:
        if state_dict["a"][classKey].size(1) != network.state_dict()[classKey].size(1):
            ratio = network.state_dict()[classKey].size(1) // state_dict["a"][classKey].size(1)
            state_dict["a"][classKey] = state_dict["a"][classKey].repeat(1,ratio)/ratio

    network.load_state_dict(state_dict['a'], strict=False)
    return network

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def norm(f):
    f = f.squeeze()
    fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
    f = f.div(fnorm.expand_as(f))
    return f

def extract_feature(model,dataloaders,writeMaps=False,dataloader_nonorm=None,grad_cam=None,grad_cam_pp=None,guided_backprop_mod=None,batch_inds=None):
    features = torch.FloatTensor()
    count = 0
    dataloader_nonorm = iter(dataloader_nonorm) if not dataloader_nonorm is None else dataloader_nonorm
    ids = get_id(image_datasets['gallery'].imgs)[1]


    for batch_idx,data in enumerate(dataloaders):

        img, label = data
        n, c, h, w = img.size()
        count += n
        if opt.use_dense:
            ff = torch.FloatTensor(n,1024).zero_()
        else:
            ff = torch.FloatTensor(n,1024).zero_()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_() # we have six parts
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())

            with torch.no_grad():
                ret = model(input_img,retSim=opt.att_maps)
            f, x = ret[:2]

            if opt.att_maps and writeMaps:
                simMaps,featNorm = ret[2],ret[3]
                if i == 0:
                    img_unorm,_ = dataloader_nonorm.next()

                if batch_idx%5 == 0 and i ==0 and (batch_inds is None or batch_idx in batch_inds):
                    #img_unorm,_ = dataloader_nonorm.next()

                    _,inds = torch.tensor(ids[batch_idx*n:(batch_idx+1)*n]).sort()

                    ids_batch = ids[batch_idx*n:(batch_idx+1)*n]

                    simMaps = simMaps[inds]
                    featNorm = featNorm[inds]
                    img_unorm = img_unorm[inds]

                    torch.save(simMaps,"../../results/{}/simMaps{}_model{}_trial{}.pt".format(opt.exp_id,batch_idx,opt.model_id,opt.which_trial))
                    torch.save(featNorm,"../../results/{}/norm{}_model{}_trial{}.pt".format(opt.exp_id,batch_idx,opt.model_id,opt.which_trial))
                    torch.save(img_unorm,"../../results/{}/img{}_model{}_trial{}.pt".format(opt.exp_id,batch_idx,opt.model_id,opt.which_trial))
                    torch.save(ids_batch,"../../results/{}/ids{}_model{}_trial{}.pt".format(opt.exp_id,batch_idx,opt.model_id,opt.which_trial))

                    if not grad_cam is None:
                        #allMask = []
                        #allMask_pp = []
                        allMap = []

                        for l in range(len(input_img)):
                            #mask = grad_cam(input_img[l:l+1])
                            #allMask.append(mask)

                            #mask_pp = grad_cam_pp(input_img[l:l+1])
                            #allMask_pp.append(mask_pp)

                            map = guided_backprop_mod.generate_gradients(input_img[l:l+1])
                            allMap.append(map)

                            grad_cam_pp.model_arch.zero_grad()

                        #allMask = torch.cat(allMask,dim=0)
                        #allMask_pp = torch.cat(allMask_pp,dim=0)
                        allMap = torch.cat(allMap,dim=0)

                        print("Writing {}",batch_idx)
                        #torch.save(allMask,"../../results/{}/gradcam{}_model{}_trial{}.pt".format(opt.exp_id,batch_idx,opt.model_id,opt.which_trial))
                        #torch.save(allMask_pp,"../../results/{}/gradcam_pp{}_model{}_trial{}.pt".format(opt.exp_id,batch_idx,opt.model_id,opt.which_trial))
                        torch.save(allMap,"../../results/{}/guidedback{}_model{}_trial{}.pt".format(opt.exp_id,batch_idx,opt.model_id,opt.which_trial))

            x[0] = norm(x[0])
            x[1] = norm(x[1])
            f = torch.cat((x[0],x[1]), dim=1) #use 512-dim feature
            f = f.data.cpu()
            ff = ff+f

        ff[:, 0:512] = norm(ff[:, 0:512])
        ff[:, 512:1024] = norm(ff[:, 512:1024])

        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)

        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs
mquery_path = image_datasets['multi-query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)
mquery_cam,mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')

###load config###
config_path = os.path.join('../outputs',name,'config.yaml')
with open("../"+opt.config, 'r') as stream:
    config = yaml.load(stream)

part_nb = config["part_nb"] if "part_nb" in config else 3

model_structure = ft_netAB(config['ID_class'], norm=config['norm_id'], stride=config['ID_stride'], pool=config['pool'],\
                                highRes=config["high_res"],nbVec=part_nb)

if opt.gradcam:
    model_structure.features = model_structure.model
    model_structure.layer4 = model_structure.model.layer4
    model_dict = dict(type="resnet", arch=model_structure, layer_name='layer4')
    grad_cam = gradcam.GradCAM(model_dict, True)
    grad_cam_pp = gradcam.GradCAMpp(model_dict,True)
    guided_backprop_mod = guided_backprop.GuidedBackprop(model_structure)
else:
    grad_cam = None
    grad_cam_pp = None
    guided_backprop_mod = None

if opt.PCB:
    model_structure = PCB(config['ID_class'])

if opt.att_maps:
    print("../../models/{}/id_[0-9]*{}model{}_trial{}_best.pt".format(opt.exp_id,opt.which_epoch,opt.model_id,opt.which_trial))
    save_path = glob.glob("../../models/{}/id_[0-9]*{}model{}_trial{}_best.pt".format(opt.exp_id,opt.which_epoch,opt.model_id,opt.which_trial))[0]
else:
    save_path = None
model = load_network(model_structure,save_path)

# Remove the final fc layer and classifier layer
model.model.fc = nn.Sequential()
model.classifier1.classifier = nn.Sequential()
model.classifier2.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
since = time.time()

if not opt.gradcam:
    with torch.no_grad():
        gallery_feature = extract_feature(model,dataloaders['gallery'],True,dataloader_nonorm,grad_cam,grad_cam_pp,guided_backprop_mod,opt.batch_inds)
else:
    gallery_feature = extract_feature(model,dataloaders['gallery'],True,dataloader_nonorm,grad_cam,grad_cam_pp,guided_backprop_mod,opt.batch_inds)

with torch.no_grad():
    query_feature = extract_feature(model,dataloaders['query'])
    time_elapsed = time.time() - since
    print('Extract features complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    if opt.multi:
        mquery_feature = extract_feature(model,dataloaders['multi-query'])

# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)
if opt.multi:
    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
    scipy.io.savemat('multi_query.mat',result)

os.system('python evaluate_gpu.py')

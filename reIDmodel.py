"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import sys

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        #num_bottleneck = input_dim # We remove the input_dim
        add_block += [nn.BatchNorm1d(num_bottleneck, affine=True)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, norm=False, pool='avg', stride=2):
        super(ft_net, self).__init__()
        if norm:
            self.norm = True
        else:
            self.norm = False
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        self.part = 4
        if pool=='max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # remove the final downsample
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)  # -> 512 32*16
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f = self.model.partpool(x) # 8 * 2048 4*1
        x = self.model.avgpool(x)  # 8 * 2048 1*1

        x = x.view(x.size(0),x.size(1))
        f = f.view(f.size(0),f.size(1)*self.part)
        if self.norm:
            fnorm = torch.norm(f, p=2, dim=1, keepdim=True) + 1e-8
            f = f.div(fnorm.expand_as(f))
        x = self.classifier(x)
        return f, x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inp):

        x = inp

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1,groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,groups=groups)

# Define the AB Model
class ft_netAB(nn.Module):

    def __init__(self, class_num, norm=False, stride=2, droprate=0.5, pool='avg',repVec=True,nbVec=3,highRes=False,teach=False,\
                        dilation=False,b_cnn=False,mergevec=False):
        super(ft_netAB, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.part = 4
        if pool=='max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.model = model_ft
        self.dilation = dilation

        if highRes:
            self.model.layer2[0].downsample[0].stride = 1
            self.model.layer2[0].conv2.stride = 1

            self.model.layer3[0].downsample[0].stride = 1
            self.model.layer3[0].conv2.stride = 1

            self.model.layer4[0].downsample[0].stride = 1
            self.model.layer4[0].conv2.stride = 1

            if dilation:

                self.model.layer2[0].conv2.dilation = 2
                self.model.layer2[0].conv2.padding = 2

                self.model.layer3[0].conv2.dilation = 2
                self.model.layer3[0].conv2.padding = 2

                self.model.layer4[0].conv2.dilation = 2
                self.model.layer4[0].conv2.padding = 2

        elif stride == 1:
            self.model.layer4[0].downsample[0].stride = (1,1)
            self.model.layer4[0].conv2.stride = (1,1)

        self.repVec = repVec
        self.nbVec = nbVec
        self.b_cnn = b_cnn

        if (not repVec) or mergevec:
            self.classifier1 = ClassBlock(2048, class_num, 0.5)
            self.classifier2 = ClassBlock(2048, class_num, 0.75)
        else:
            print("nbVec",nbVec)
            self.classifier1 = ClassBlock(2048*nbVec, class_num, 0.5)
            self.classifier2 = ClassBlock(2048*nbVec, class_num, 0.75)

            if self.b_cnn:
                attention = [BasicBlock(2048, 2048)]
                attention.append(conv1x1(2048, nbVec))
                attention.append(nn.ReLU())
                self.att = nn.Sequential(*attention)

        self.teach = teach
        self.mergevec = mergevec
        if self.mergevec:
            half = nn.Linear(2048,2048//2)
            quarter = nn.Linear(2048,2048//4)
            self.vec1 = half
            self.vec2 = quarter
            self.vec3 = quarter

    def compAtt(self,x):
        attMaps = self.att(x)
        x = (attMaps.unsqueeze(2)*x.unsqueeze(1)).reshape(x.size(0),x.size(1)*(attMaps.size(1)),x.size(2),x.size(3))
        x = self.model.avgpool(x)
        return x.view(x.size(0), -1),[attMaps[:,i:i+1] for i in range(attMaps.size(1))]

    def mergeVec(self,x):
        finalVec = []
        for k,vec in enumerate(x):
            lin = getattr(self,"vec{}".format(k+1))
            outVec = lin(vec)
            finalVec.append(outVec)
        x = torch.cat(finalVec,dim=-1)
        return x

    def forward(self, x,retSim=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f = self.model.partpool(x)

        f = f.view(f.size(0),f.size(1)*self.part)
        f = f.detach() # no gradient

        if not self.repVec:
            x = self.model.avgpool(x)
        else:
            if retSim:
                norm = torch.sqrt(torch.pow(x,2).sum(dim=1,keepdim=True))

            if self.b_cnn:
                x,simMaps = self.compAtt(x)
            else:
                x,simMaps = representativeVectors(x,self.nbVec)

                if self.mergevec:
                    x = self.mergeVec(x)
                else:
                    x = torch.cat(x,dim=1)

        x = x.view(x.size(0), x.size(1))
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x=[]
        x.append(x1)
        x.append(x2)

        if self.teach:
            return f,x[0]
        else:
            if not retSim:
                return f, x
            else:
                return f,x,simMaps,norm

def representativeVectors(x,nbVec):

    xOrigShape = x.size()

    x = x.permute(0,2,3,1).reshape(x.size(0),x.size(2)*x.size(3),x.size(1))
    norm = torch.sqrt(torch.pow(x,2).sum(dim=-1)) + 0.00001

    raw_reprVec_score = norm.clone()

    repreVecList = []
    simList = []
    for _ in range(nbVec):
        _,ind = raw_reprVec_score.max(dim=1,keepdim=True)
        raw_reprVec_norm = norm[torch.arange(x.size(0)).unsqueeze(1),ind]
        raw_reprVec = x[torch.arange(x.size(0)).unsqueeze(1),ind]
        sim = (x*raw_reprVec).sum(dim=-1)/(norm*raw_reprVec_norm)
        simNorm = sim/sim.sum(dim=1,keepdim=True)
        reprVec = (x*simNorm.unsqueeze(-1)).sum(dim=1)
        repreVecList.append(reprVec)
        raw_reprVec_score = (1-sim)*raw_reprVec_score
        simReshaped = simNorm.reshape(sim.size(0),1,xOrigShape[2],xOrigShape[3])
        simList.append(simReshaped)

    simList = torch.cat(simList,dim=1)
    return repreVecList,simList


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num ):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num ):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num ):
        super(PCB, self).__init__()

        self.part = 4 # We cut the pool5 to 4 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        self.softmax = nn.Softmax(dim=1)
        # define 4 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, True, False, 256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        f = x
        f = f.view(f.size(0),f.size(1)*self.part)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get part feature batchsize*2048*4
        for i in range(self.part):
            part[i] = x[:,:,i].contiguous()
            part[i] = part[i].view(x.size(0), x.size(1))
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        y=[]
        for i in range(self.part):
            y.append(predict[i])

        return f, y

class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        # remove the final downsample
        self.model.layer3[0].downsample[0].stride = (1,1)
        self.model.layer3[0].conv2.stride = (1,1)

        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2))
        return y

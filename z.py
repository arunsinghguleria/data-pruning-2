import torchvision
import torch
import torch.nn as nn
from torch import optim
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm

transformations=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
trainset=torchvision.datasets.CIFAR10(root='./CIFAR10',download=True,transform=transformations,train=True)

testset=torchvision.datasets.CIFAR10(root='./CIFAR10',download=True,transform=transformations,train=False)

trainloader=DataLoader(dataset=trainset,batch_size=16)
testloader=DataLoader(dataset=testset,batch_size=16)

inputs,labels=next(iter(trainloader))
labels=labels.float()
inputs.size()

print(labels.type())
resnet=torchvision.models.resnet50(pretrained=True)

if torch.cuda.is_available():
  resnet=resnet.cuda(2)
  inputs,labels=inputs.cuda(2),torch.Tensor(labels).cuda(2)

outputs=resnet(inputs)
outputs.size()

for param in resnet.parameters():
  param.requires_grad=False

numft=resnet.fc.in_features
print(numft)
resnet.fc=torch.nn.Sequential(nn.Linear(numft,1000),nn.ReLU(),nn.Linear(1000,10))
resnet.cuda(2)
resnet.train(True)
optimizer=torch.optim.SGD(resnet.parameters(),lr=1e-4,momentum=0.9)
criterion=nn.CrossEntropyLoss()

for epoch in range(100):
    resnet.train(True)

    trainloss=0
    correct=0
    for x,y in tqdm(trainloader):
        x,y=x.cuda(2),y.cuda(2)
        optimizer.zero_grad()
        
        yhat=resnet(x)
        loss=criterion(yhat,y)
        
        loss.backward()
        optimizer.step()
        trainloss+=loss.item()
        
        
    
    
    accuracy=[]
    running_corrects=0.0
    for x_test,y_test in tqdm(testloader):
        
        x_test,y_test=x_test.cuda(2),y_test.cuda(2)
        yhat=resnet(x_test)
        _,z=yhat.max(1)
        running_corrects += torch.sum(y_test == z)
    
    accuracy.append(running_corrects/len(testset))
    print('Epoch: {} Loss: {} testAccuracy: {}'.format(epoch,(trainloss/len(trainset)),running_corrects/len(testset)))

print(running_corrects/len(testset))
# accuracy=max(accuracy)
print(accuracy)
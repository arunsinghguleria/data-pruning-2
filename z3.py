from utils import DenseNet121_Multi_Class, Inception_Multi_Class, ResNet_Multi_Class, ResNeXt_Multi_Class, calculate_classwise_accuracy_CIFAR, get_scores,load_config


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
from dataset_Generator import DataSetGenerator



class_count = 10


config = load_config('hyperparameters/cifar10/0cifar10.yaml')

model = ResNeXt_Multi_Class(classCount=class_count)            

device = 'cuda:1'

model = model.to(device)

transformations=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

trainset=torchvision.datasets.CIFAR10(root='./CIFAR10',download=True,transform=transformations,train=True)
testset=torchvision.datasets.CIFAR10(root='./CIFAR10',download=True,transform=transformations,train=False)

trainloader=DataLoader(dataset=trainset,batch_size=64)
testloader=DataLoader(dataset=testset,batch_size=64)


train_config = config.get('cifar10',None).train
test_config = config.get('cifar10',None).test
trainset =  DataSetGenerator(train_config)
testset =  DataSetGenerator(test_config)
sampler = None


trainloader =  DataLoader(dataset = trainset,
                                    batch_size = config.batch_size,
                                    # shuffle = True, commented due to sampler (both are mutually exclusive)
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=False, # earlier was true
                                    # sampler = sampler,
                                    shuffle = True

                                    )
    
testloader =  DataLoader(dataset = testset,
                                    batch_size = config.batch_size,
                                    shuffle = True,
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=False # earlier was true
                                    )




# optimizer=torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay=0)

criterion = nn.CrossEntropyLoss()


for epoch in range(100):
    model.train(True)

    trainloss=0
    correct=0
    for x,y in tqdm(trainloader):
        # print(x.shape,y.shape,type(x),type(y))
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        
        yhat=model(x)
        loss=criterion(yhat,y)
        
        loss.backward()
        optimizer.step()
        trainloss+=loss.item()
        
        
    
    
    accuracy=[]
    running_corrects=0.0
    for x_test,y_test in tqdm(testloader):
        
        x_test,y_test=x_test.to(device),y_test.to(device)
        yhat=model(x_test)
        _,z=yhat.max(1)
        running_corrects += torch.sum(y_test == z)
    
    accuracy.append(running_corrects/len(testset))
    print('Epoch: {} Loss: {} testAccuracy: {}'.format(epoch,(trainloss/len(trainset)),running_corrects/len(testset)))

print(running_corrects/len(testset))
# accuracy=max(accuracy)
print(accuracy)
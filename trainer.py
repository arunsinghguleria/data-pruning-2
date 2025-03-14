'''


set shuffle to true in dataloader

python3 trainer.py --exp_name exp0 --dataset cifar10 --config hyperparameters/cifar10.yaml --device cuda:1
python3 trainer.py --exp_name exp1 --dataset nih --config hyperparameters/nih.yaml --device cuda:1



python3 trainer.py --exp_name 1cifar --dataset cifar10 --config hyperparameters/cifar10/1cifar10.yaml --device cuda:1
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import DenseNet121_Multi_Class, Inception_Multi_Class, ResNet_Multi_Class, ResNeXt_Multi_Class, calculate_classwise_accuracy_CIFAR, get_scores
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms


import argparse
from utils import load_config
import os
import pandas as pd

from dataset_Generator import DataSetGenerator





def train(args,config):
    train_config = config.get(args.dataset,None).train
    test_config = config.get(args.dataset,None).test
    train_data =  DataSetGenerator(train_config)
    test_data =  DataSetGenerator(test_config)

    df = {}
    if(train_config.cifar_sample):
        df['sample'] = '-'.join(map(str,train_config.cifar_sample))
    if(train_config.cifar_prune_ratio):
        df['prune_ratio'] = '-'.join(map(str,train_config.cifar_prune_ratio))


        
    

    
    device = args.device


    num_samples = sum(train_data.class_wise_count)
    class_weights = [1/train_data.class_wise_count[i] for i in range(len(train_data.class_wise_count))]
    weights = [class_weights[i] for i in train_data.labels]


    if(config.use_sampler):
        print('using sampler')
        args.exp_name = args.exp_name + '_sampling_used_'
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights),int(num_samples))
        shuffle = False
    else:
        sampler = None
        shuffle = True

    if(train_config.prune_file):
        args.exp_name = args.exp_name + f'_{train_config.prune_file.split('/')[-1]}_'
    if(train_config.use_augmented_data):
        args.exp_name = args.exp_name + '_use_augmented_data_'



    train_data_loader =  DataLoader(dataset = train_data,
                                    batch_size = config.batch_size,
                                    # shuffle = True, commented due to sampler (both are mutually exclusive)
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=False, # earlier was true
                                    sampler = sampler,
                                    shuffle = shuffle

                                    )
    
    test_data_loader =  DataLoader(dataset = test_data,
                                    batch_size = config.batch_size,
                                    shuffle = True,
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=False # earlier was true
                                    )
    

    transformations=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    trainset=torchvision.datasets.CIFAR10(root='./CIFAR10',download=True,transform=transformations,train=True)
    testset=torchvision.datasets.CIFAR10(root='./CIFAR10',download=True,transform=transformations,train=False)

    trainloader=DataLoader(dataset=trainset,batch_size=64)
    testloader=DataLoader(dataset=testset,batch_size=64)


    if(config.use_weighted_loss):
        print('using weighted loss and weights are - ',torch.tensor(class_weights))
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights,device=device))
    elif(config.use_class_balanced_loss):
        class_weights = [(1-config.beta)/(1-(config.beta**train_data.class_wise_count[i])) for i in range(len(train_data.class_wise_count))]
        print('using class_balanced_loss loss and weights are - ',torch.tensor(class_weights))
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights,device=device))

    else:
        criterion = nn.CrossEntropyLoss()

    if args.model== "densenet":
        model = DenseNet121_Multi_Class(classCount=config.class_count)
    elif args.model== "inception":
        model = Inception_Multi_Class(classCount=config.class_count)
    elif args.model== "resnet":
        model = ResNet_Multi_Class(classCount=config.class_count)    
    elif args.model== "resnext":
        model = ResNeXt_Multi_Class(classCount=config.class_count)            
        # print(model)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.7, patience=5)

    scheduler = ReduceLROnPlateau(optimizer, patience=4, verbose=True, factor=0.2)
    
    print(config.batch_size)
    # iter_epochs = tqdm(range(args.num_epochs),ncols=50,desc='Train Loss {} Test : Loss {}, Accuracy {}'.format(train_loss, test_loss, test_accuracy))
    max_valid_acc = 0.0
    training_samples_cnt = [0] * config.class_count

    train_loss = 0
    test_loss = 0
    test_accuracy = 0

    # EL2N_score = pd.DataFrame(list(range(train_data.n)), columns=['image_name'])
    # GraNd_score = pd.DataFrame(list(range(train_data.n)), columns=['image_name'])
    EL2N_score = pd.DataFrame()
    GraNd_score = pd.DataFrame()

    path = f'{config.results_folder}{args.exp_name}.csv'
    
    if(os.path.isfile(path)):
        print('file exist, results will be appended.')
        data_frame = pd.read_csv(path)
    else:
        print('file doesn\'t exist, new file will be created.')
        data_frame = pd.DataFrame()


    for epoch_num in tqdm(range(config.num_epochs),ncols=50):
        df['epoch_num'] = epoch_num
        true_class = torch.zeros(10)
        predict_class = torch.zeros(10)
        train_losses = 0
        train_correct = 0
        valid_losses = 0
        valid_correct = 0
        test_losses = 0
        test_correct = 0
        training_samples_cnt = [0] * config.class_count

        # training mode
        model.train()
        for batch_no, (images, labels) in enumerate(train_data_loader):
            images = images.to(device)
            labels = labels.to(device)
            # zeroing the optimizer
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)   
            
            _, prediction = outputs.max(1)

            if epoch_num == 0: # or True:
                for i in range(config.class_count):
                    training_samples_cnt[i] += torch.sum(labels==i).item()
            
            train_losses += loss.item()
            loss.backward()
            optimizer.step()

            train_correct += (prediction == labels).sum()

        train_loss = train_losses / len(train_data_loader)
 

        true_class = torch.zeros(10)
        predict_class = torch.zeros(10)

        if(config.get_scores):
            get_scores(model,train_data,optimizer,criterion,device,EL2N_score,GraNd_score,epoch_num,config.path_to_save_score)
            continue
        model.eval()
        for batch_no, (images, labels) in enumerate(test_data_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
           
            _, prediction = outputs.max(1)

            loss = criterion(outputs, labels)

            for i in range(10):
                true_class[i] += torch.sum(labels==i).item()
                predict_class[i] += torch.sum((labels==i) & (labels == prediction)).item()

            
            test_losses += loss.item()

            test_correct += (prediction == labels).sum()
            
        test_accuracy = test_correct.item() / test_data.n
        test_loss = test_losses / test_data.n
        acc = torch.div(predict_class,true_class)

        epoch_stats = {}
        for i in range(10):
            epoch_stats['test_class_acc_'+str(i)] = acc[i].item()
            df['test_class_acc_'+str(i)] = acc[i].item()

        df['test_acc'] = test_accuracy
        
        # training_samples_cnt
        li = [i for i in zip(training_samples_cnt,acc.tolist())]
        li = sorted(li, key = lambda i: i[0],reverse=True)

        accuracy = calculate_classwise_accuracy_CIFAR(li)

        epoch_stats['test_head_acc'] = accuracy['head']
        epoch_stats['test_medium_acc'] = accuracy['medium']
        epoch_stats['test_tail_acc'] = accuracy['tail']
        epoch_stats['test_total_acc'] = accuracy['total']
        
        epoch_stats['train_loss'] = train_loss 
        # epoch_stats['validation_loss'] = valid_loss
        epoch_stats['test_loss'] = test_loss


        df['test_head_acc'] = accuracy['head']
        df['test_medium_acc'] = accuracy['medium']
        df['test_tail_acc'] = accuracy['tail']
        df['test_total_acc'] = accuracy['total']
        
        df['train_loss'] = train_loss 
        # df['validation_loss'] = valid_loss
        df['test_loss'] = test_loss


        scheduler.step(train_loss)

        new_row = pd.DataFrame([df])
        data_frame = pd.concat([data_frame,new_row],ignore_index=True)
    
    if(config.get_scores == False):
        data_frame.to_csv(f'{config.results_folder}{args.exp_name}.csv', index=False)
        print(f'Results saved in {config.results_folder}{args.exp_name}.csv')
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp-0',required=True, help='path to your yaml config file')
    parser.add_argument('--config', type=str, default='hyperparameters/cifar10.yaml', help='path to your yaml config file')
    parser.add_argument('--dataset', type=str, default='cifar10',choices=['cifar10','nih'], help='name of your dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device name and id')
    parser.add_argument('--model', type=str, default='resnext', help='model to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 regularization.')



    args = parser.parse_args()

    config = load_config(args.config)

    train(args,config)

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
#from torcheval.metrics.functional import multiclass_f1_score
import os
import numpy as np
import random
import pandas as pd


# In[2]:
# Input Params
train_data = 'medium_data_path'
num_epoch = 100
learning_rate = 0.001
lr_step = 10

data_path = '../Data/full dataset'
medium_data_path = '../Data/Small Dataset 1000'
extra_small_data_path = '../Data/Small Dataset 200'
figures_output_path = '../Outputs/figures'
csv_outputs ='../Outputs/csv'
models_output_path = '../Models'
model_checkpoints_path = '../Models/checkpoints'



datasetsizes ={'extra_small_data_path':200, 'medium_data_path':1000}



# If output folder exists, dont create a new one, if does not exits then create it
try:
    os.listdir(os.path.join(model_checkpoints_path, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}"))
except:
    os.mkdir(os.path.join(model_checkpoints_path, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}"))

try:
    os.listdir(os.path.join(csv_outputs, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}"))
except:
    os.mkdir(os.path.join(csv_outputs,f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}"))


load_checkpoints=False


# Create transforms
data_transforms = {'train': transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'val': transforms.Compose([
                      transforms.Resize((300, 300)),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'test': transforms.Compose([
                      transforms.Resize((300,300)),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}




# Create datasets
image_datasets = {x:datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x]) for x in [ 'val', 'test']}
image_datasets['train'] = datasets.ImageFolder(os.path.join(medium_data_path,'train'), data_transforms['train'])



# Data loaders
dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True) for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes


print(f"Classes in the dataset are:{class_names}")

print(f"Num batches in training dataset:{len(dataloaders['train'])}")
print(f"Num images in training dataset:{dataset_sizes['train']}")

print(f"Num batches in val dataset:{len(dataloaders['val'])}")
print(f"Num images in val dataset:{dataset_sizes['val']}")

print(f"Num batches in test dataset:{len(dataloaders['test'])}")
print(f"Num images in test dataset:{dataset_sizes['test']}")
print(f"GPU: {torch.cuda.get_device_name(0)}")



###############################
# Resnet 18 

###############################

model_conv = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
# Dont store gradients in the pretrained layers
for param in model_conv.parameters():
    param.required_grad=False
# Replace final layer with new one with 5 output nodes
n_inputs = model_conv.fc.in_features
model_conv.fc= nn.Linear(in_features=n_inputs, out_features=len(class_names))

if load_checkpoints==True:
	checkpoint = torch.load(os.path.join(model_checkpoints_path, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}",f'ResNet18_{datasetsizes[train_data]}.pt' ))
	model_conv.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
else:
	pass


# move to GPU if available
if torch.cuda.is_available():
    model_conv.cuda()
    
# Set up loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_conv.fc.parameters(), lr=learning_rate, momentum=0.9)
exp_lr_sch = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)



train_loss=[]
train_accuracy=[]
val_loss=[]
val_accuracy=[]

for epoch in range(num_epoch):
    exp_lr_sch.step()
    iterations=0
    iter_loss=0.0
    correct=0

    model_conv.train()

    for images, labels in dataloaders['train']:
        images = Variable(images)
        labels=Variable(labels)
        if torch.cuda.is_available():
            images=images.cuda()
            labels=labels.cuda()

        optimizer.zero_grad()
        outputs= model_conv(images)
        loss=criterion(outputs, labels)
        iter_loss+=loss.item()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs,1)
        correct+=(predicted==labels).sum()
        iterations+=1

    train_loss.append(iter_loss/iterations)
    train_iter_acc = 100*correct/dataset_sizes['train']
    train_accuracy.append(train_iter_acc)
    print(f"Epoch {epoch+1}/{num_epoch}, Loss {loss.item()}, train accuracy {train_iter_acc}")
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model_conv.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, os.path.join(model_checkpoints_path, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}", f'ResNet18_{datasetsizes[train_data]}.pt'))

    if epoch%5==0:
        model_conv.eval()
        test_loss=0.0
        correct=0
        iterations=0

        for images, labels in dataloaders['val']:
            images = Variable(images)
            labels=Variable(labels)
            if torch.cuda.is_available():
                images=images.cuda()
                labels=labels.cuda()


            outputs = model_conv(images)
            loss=criterion(outputs, labels)
            iter_loss+=loss.item()
            _, predicted = torch.max(outputs,1)
            correct+=(predicted==labels).sum()
            iterations+=1

        val_acc = 100*correct/dataset_sizes['val']
        print(f"ResNet18 - Epoch {epoch+1}/{num_epoch}, Loss {loss.item()}, val accuracy {val_acc}")
        val_loss.append(iter_loss/iterations)
        val_iter_acc = 100*correct/dataset_sizes['val']
        val_accuracy.append(val_iter_acc)
    else:
        pass
train_accuracy = [i.item() for i in train_accuracy]
val_accuracy = [i.item() for i in val_accuracy]
pd.DataFrame([train_loss, train_accuracy, val_loss, val_accuracy]).to_excel(os.path.join(csv_outputs, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}",f'ResNet18_{datasetsizes[train_data]}_stats.xlsx'))
pd.DataFrame([predicted, labels]).to_excel(os.path.join(csv_outputs, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}", f'ResNet18_{datasetsizes[train_data]}_outputs.xlsx'))


###############################
# ResNet 50
# Use V2 weights as they are an improvement over V1 weights
###############################


model_conv = torchvision.models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
# Dont store gradients in the pretrained layers
for param in model_conv.parameters():
    param.required_grad=False
# Replace final layer with new one with 5 output nodes
n_inputs = model_conv.fc.in_features
model_conv.fc= nn.Linear(in_features=n_inputs, out_features=len(class_names))

if load_checkpoints==True:
	checkpoint = torch.load(os.path.join(model_checkpoints_path, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}",f'ResNet50_{datasetsizes[train_data]}.pt' ))
	model_conv.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
else:
	pass

# move to GPU if available
if torch.cuda.is_available():
    model_conv.cuda()
    
    
# Set up loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_conv.fc.parameters(), lr=learning_rate, momentum=0.9)
exp_lr_sch = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)



train_loss=[]
train_accuracy=[]
val_loss=[]
val_accuracy=[]

for epoch in range(num_epoch):
    exp_lr_sch.step()
    iterations=0
    iter_loss=0.0
    correct=0

    model_conv.train()

    for images, labels in dataloaders['train']:
        images = Variable(images)
        labels=Variable(labels)
        if torch.cuda.is_available():
            images=images.cuda()
            labels=labels.cuda()

        optimizer.zero_grad()
        outputs= model_conv(images)
        loss=criterion(outputs, labels)
        iter_loss+=loss.item()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs,1)
        correct+=(predicted==labels).sum()
        iterations+=1

    train_loss.append(iter_loss/iterations)
    train_iter_acc = 100*correct/dataset_sizes['train']
    train_accuracy.append(train_iter_acc)
    print(f"Epoch {epoch+1}/{num_epoch}, Loss {loss.item()}, train accuracy {train_iter_acc}")
    torch.save({
    	'epoch': epoch,
	'model_state_dict': model_conv.state_dict(),
	'optimizer_state_dict': optimizer.state_dict(),
	'loss': loss,
	}, os.path.join(model_checkpoints_path, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}", f'ResNet50_{datasetsizes[train_data]}.pt'))

    if epoch%5==0:
        model_conv.eval()
        test_loss=0.0
        correct=0
        iterations=0

        for images, labels in dataloaders['val']:
            images = Variable(images)
            labels=Variable(labels)
            if torch.cuda.is_available():
                images=images.cuda()
                labels=labels.cuda()


            outputs = model_conv(images)
            loss=criterion(outputs, labels)
            iter_loss+=loss.item()
            _, predicted = torch.max(outputs,1)
            correct+=(predicted==labels).sum()
            iterations+=1

        val_acc = 100*correct/dataset_sizes['val']
        print(f"Resnet50 - Epoch {epoch+1}/{num_epoch}, Loss {loss.item()}, val accuracy {val_acc}")
        val_loss.append(iter_loss/iterations)
        val_iter_acc = 100*correct/dataset_sizes['val']
        val_accuracy.append(val_iter_acc)
    else:
        pass


train_accuracy = [i.item() for i in train_accuracy]
val_accuracy = [i.item() for i in val_accuracy]
pd.DataFrame([train_loss, train_accuracy, val_loss, val_accuracy]).to_excel(os.path.join(csv_outputs, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}", f'ResNet50_{datasetsizes[train_data]}_stats.xlsx'))
pd.DataFrame([predicted, labels]).to_excel(os.path.join(csv_outputs, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}", f'ResNet50_{datasetsizes[train_data]}_outputs.xlsx'))






###############################
# Inception V3

###############################


#load model

model_conv = torchvision.models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')

# Freeze layers in the model to prevent disturbing the weights
for param in model_conv.parameters():
    param.required_grad=False


# Replace final layer with new one with 5 output nodes
n_inputs = model_conv.fc.in_features
model_conv.fc= nn.Linear(in_features=n_inputs, out_features=len(class_names))

if load_checkpoints==True:
	checkpoint = torch.load(os.path.join(model_checkpoints_path, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}",f'InceptionV3_{datasetsizes[train_data]}.pt' ))
	model_conv.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
else:
	pass


if torch.cuda.is_available():
    model_conv.cuda()



    # Set up loss function and optimiser

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_conv.fc.parameters(), lr=learning_rate, momentum=0.9)

    exp_lr_sch = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)

    train_loss=[]
    train_accuracy=[]
    val_loss=[]
    val_accuracy=[]
    
    train_loss=[]
    train_accuracy=[]
    val_loss=[]
    val_accuracy=[]

    for epoch in range(num_epoch):
        exp_lr_sch.step()
        iterations=0
        iter_loss=0.0
        correct=0

        model_conv.train()

        for images, labels in dataloaders['train']:
            images = Variable(images)
            labels=Variable(labels)
            if torch.cuda.is_available():
                images=images.cuda()
                labels=labels.cuda()

            optimizer.zero_grad()
            outputs,_= model_conv(images)
            loss=criterion(outputs, labels)
            iter_loss+=loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs,1)
            correct+=(predicted==labels).sum()
            iterations+=1

        train_loss.append(iter_loss/iterations)
        train_iter_acc = 100*correct/dataset_sizes['train']
        train_accuracy.append(train_iter_acc)
        print(f"Epoch {epoch+1}/{num_epoch}, Loss {loss.item()}, train accuracy {train_iter_acc}")

        torch.save({
                'epoch': epoch,
                'model_state_dict': model_conv.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, os.path.join(model_checkpoints_path, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}", f'InceptionV3_{datasetsizes[train_data]}.pt'))
        if epoch%5==0:
            model_conv.eval()
            test_loss=0.0
            correct=0
            iterations=0

            for images, labels in dataloaders['val']:
                images = Variable(images)
                labels=Variable(labels)
                if torch.cuda.is_available():
                    images=images.cuda()
                    labels=labels.cuda()


                outputs= model_conv(images)
                loss=criterion(outputs, labels)
                iter_loss+=loss.item()
                _, predicted = torch.max(outputs,1)
                correct+=(predicted==labels).sum()
                iterations+=1

            val_acc = 100*correct/dataset_sizes['val']
            print(f"InceptionV3 - Epoch {epoch+1}/{num_epoch}, Loss {loss.item()}, val accuracy {val_acc}")
            val_loss.append(iter_loss/iterations)
            val_iter_acc = 100*correct/dataset_sizes['val']
            val_accuracy.append(val_iter_acc)
        else:
            pass





    train_accuracy = [i.item() for i in train_accuracy]
    val_accuracy = [i.item() for i in val_accuracy]
    pd.DataFrame([train_loss, train_accuracy, val_loss, val_accuracy]).to_excel(os.path.join(csv_outputs, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}", f'InceptionV3_{datasetsizes[train_data]}_stats.xlsx'))
    pd.DataFrame([predicted, labels]).to_excel(os.path.join(csv_outputs, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}", f'InceptionV3_{datasetsizes[train_data]}_outputs.xlsx'))


    ###############################
    # VGG16 

    ###############################

    model_conv = torchvision.models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
    # Dont store gradients in the pretrained layers
    for param in model_conv.parameters():
        param.required_grad=False
    # Replace final layer with new one with 5 output nodes


    model_conv.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names))

    if load_checkpoints==True:
        checkpoint = torch.load(os.path.join(model_checkpoints_path, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}",f'VGG16_{datasetsizes[train_data]}.pt' ))
        model_conv.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        pass


    # move to GPU if available
    if torch.cuda.is_available():
        model_conv.cuda()
        
    # Set up loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_conv.classifier.parameters(), lr=learning_rate, momentum=0.9)
    exp_lr_sch = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)



    train_loss=[]
    train_accuracy=[]
    val_loss=[]
    val_accuracy=[]

    for epoch in range(num_epoch):
        exp_lr_sch.step()
        iterations=0
        iter_loss=0.0
        correct=0

        model_conv.train()

        for images, labels in dataloaders['train']:
            images = Variable(images)
            labels=Variable(labels)
            if torch.cuda.is_available():
                images=images.cuda()
                labels=labels.cuda()

            optimizer.zero_grad()
            outputs= model_conv(images)
            loss=criterion(outputs, labels)
            iter_loss+=loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs,1)
            correct+=(predicted==labels).sum()
            iterations+=1

        train_loss.append(iter_loss/iterations)
        train_iter_acc = 100*correct/dataset_sizes['train']
        train_accuracy.append(train_iter_acc)
        print(f"Epoch {epoch+1}/{num_epoch}, Loss {loss.item()}, train accuracy {train_iter_acc}")
        
        torch.save({
                'epoch': epoch,
                'model_state_dict': model_conv.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, os.path.join(model_checkpoints_path, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}", f'VGG16_{datasetsizes[train_data]}.pt'))

        if epoch%5==0:
            model_conv.eval()
            test_loss=0.0
            correct=0
            iterations=0

            for images, labels in dataloaders['val']:
                images = Variable(images)
                labels=Variable(labels)
                if torch.cuda.is_available():
                    images=images.cuda()
                    labels=labels.cuda()


                outputs = model_conv(images)
                loss=criterion(outputs, labels)
                iter_loss+=loss.item()
                _, predicted = torch.max(outputs,1)
                correct+=(predicted==labels).sum()
                iterations+=1

            val_acc = 100*correct/dataset_sizes['val']
            print(f"VGG16 - Epoch {epoch+1}/{num_epoch}, Loss {loss.item()}, val accuracy {val_acc}")
            val_loss.append(iter_loss/iterations)
            val_iter_acc = 100*correct/dataset_sizes['val']
            val_accuracy.append(val_iter_acc)
        else:
            pass
    train_accuracy = [i.item() for i in train_accuracy]
    val_accuracy = [i.item() for i in val_accuracy]
    pd.DataFrame([train_loss, train_accuracy, val_loss, val_accuracy]).to_excel(os.path.join(csv_outputs, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}",f'VGG16_{datasetsizes[train_data]}_stats.xlsx'))
    pd.DataFrame([predicted, labels]).to_excel(os.path.join(csv_outputs, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}", f'VGG16_{datasetsizes[train_data]}_outputs.xlsx'))
    os.system('sudo shutdown 5 -h')

    ###############################
    # EfficientNet Small
    # Small model is used for GPU RAM constraints
    ###############################
    # Reduce batch size for GPU RAM constraint
    dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True) for x in ['train', 'val', 'test']}

    
    model_conv = torchvision.models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.IMAGENET1K_V1')
    # Dont store gradients in the pretrained layers
    for param in model_conv.parameters():
        param.required_grad=False
    # Replace final layer with new one with 5 output nodes
    model_conv.classifier[-1] = nn.Linear(in_features=1280, out_features=len(class_names))

    if load_checkpoints==True:
        checkpoint = torch.load(os.path.join(model_checkpoints_path, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}",f'EfficientNet_{datasetsizes[train_data]}.pt' ))
        model_conv.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        pass

    # move to GPU if available
    if torch.cuda.is_available():
        model_conv.cuda()
        
        
    # Set up loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_conv.classifier.parameters(), lr=learning_rate, momentum=0.9)
    exp_lr_sch = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)



    train_loss=[]
    train_accuracy=[]
    val_loss=[]
    val_accuracy=[]

    for epoch in range(num_epoch):
        exp_lr_sch.step()
        iterations=0
        iter_loss=0.0
        correct=0

        model_conv.train()

        for images, labels in dataloaders['train']:
            images = Variable(images)
            labels=Variable(labels)
            if torch.cuda.is_available():
                images=images.cuda()
                labels=labels.cuda()

            optimizer.zero_grad()
            outputs= model_conv(images)
            loss=criterion(outputs, labels)
            iter_loss+=loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs,1)
            correct+=(predicted==labels).sum()
            iterations+=1

        train_loss.append(iter_loss/iterations)
        train_iter_acc = 100*correct/dataset_sizes['train']
        train_accuracy.append(train_iter_acc)
        print(f"Epoch {epoch+1}/{num_epoch}, Loss {loss.item()}, train accuracy {train_iter_acc}")
        torch.save({
            'epoch': epoch,
        'model_state_dict': model_conv.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, os.path.join(model_checkpoints_path, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}", f'EfficientNet_{datasetsizes[train_data]}.pt'))

        if epoch%5==0:
            model_conv.eval()
            test_loss=0.0
            correct=0
            iterations=0

            for images, labels in dataloaders['val']:
                images = Variable(images)
                labels=Variable(labels)
                if torch.cuda.is_available():
                    images=images.cuda()
                    labels=labels.cuda()


                outputs = model_conv(images)
                loss=criterion(outputs, labels)
                iter_loss+=loss.item()
                _, predicted = torch.max(outputs,1)
                correct+=(predicted==labels).sum()
                iterations+=1

            val_acc = 100*correct/dataset_sizes['val']
            print(f"EfficientNet - Epoch {epoch+1}/{num_epoch}, Loss {loss.item()}, val accuracy {val_acc}")
            val_loss.append(iter_loss/iterations)
            val_iter_acc = 100*correct/dataset_sizes['val']
            val_accuracy.append(val_iter_acc)
        else:
            pass


    train_accuracy = [i.item() for i in train_accuracy]
    val_accuracy = [i.item() for i in val_accuracy]
    pd.DataFrame([train_loss, train_accuracy, val_loss, val_accuracy]).to_excel(os.path.join(csv_outputs, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}", f'EfficientNet_{datasetsizes[train_data]}_stats.xlsx'))
    pd.DataFrame([predicted, labels]).to_excel(os.path.join(csv_outputs, f'Dataset {datasetsizes[train_data]}',f"LR_{learning_rate}",f'EfficientNet_{datasetsizes[train_data]}_outputs.xlsx'))





else:
    print("Issue with cuda")

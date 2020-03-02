import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms,models
import matplotlib.pyplot as plt
from torch import nn,optim
import json
from utility_fun import load_dataset
from PIL import Image
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Training process is going on')
parser.add_argument('data_directory',action='store',help='Training data location')
parser.add_argument('--save_dir',action='store',dest='save_checkpoint',help='Location of tranined model/checkpoint',default='checkpoint.pth')
parser.add_argument('--arch',action='store',dest='pretrained_model',help='Pretrained Model Architecture',default='vgg16')
parser.add_argument('--learning_rate',action='store',dest='learning_rate',help='Learning Rate',default='0.001')
parser.add_argument('--dropout',action='store',dest='dropout',help='Add dropout probability',default='0.04')
parser.add_argument('--hidden_units',action='store',dest='hidden_units',help='Hidden Units',default='256')
parser.add_argument('--epochs',action='store',dest='epoch',help='No of Epochs',default='5')
parser.add_argument('--gpu',action='store',dest='gpu',help='Device on GPU?',default='on')

args = parser.parse_args()

data_directory=args.data_directory
checkpoint_path=args.save_checkpoint
pretrained_model=args.pretrained_model
learning_rate=float(args.learning_rate)
dropout=float(args.dropout)
hidden_units=int(args.hidden_units)
epochs=int(args.epoch)
gpu=args.gpu
save_dir=args.save_checkpoint


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


   
#freeze parameters
model = getattr(models,pretrained_model)(pretrained=True)

for param in model.parameters():
    param.requires_grad=False

if pretrained_model=='vgg16':
    classifier=nn.Sequential(OrderedDict([
                            ('fc1',nn.Linear(25088,hidden_units)),
                            ('relu1',nn.ReLU()),
                            ('dropout',nn.Dropout(0.4)), 
                            ('fc2',nn.Linear(hidden_units,102)),
                            ('output',nn.LogSoftmax(dim=1))]))
    model.classifier=classifier
elif pretrained_model=='densenet121':
    classifier=nn.Sequential(OrderedDict([
                            ('fc1',nn.Linear(1024,hidden_units)),
                            ('relu1',nn.ReLU()),
                            ('dropout',nn.Dropout(0.4)),     
                            ('fc2',nn.Linear(hidden_units,102)),
                            ('output',nn.LogSoftmax(dim=1))]))
    model.classifier=classifier                            

optimizer= optim.Adam(model.classifier.parameters(),lr=learning_rate)

if gpu=='on':
    device='cuda'
else:
    device='cpu'
trainloader,validloader,testloader,train_data=load_dataset(data_directory)

def get_pretrained_model():
    return pretrained_model

model.to(device)
steps=0
running_loss=0
print_every=40
criterion = nn.NLLLoss()
from workspace_utils import active_session

with active_session():

    for epoch in range(epochs):
        for inputs,labels in trainloader:
            steps=steps+1
        
            #move inputs and labels to default device
            inputs,labels=inputs.to(device),labels.to(device)
        
            optimizer.zero_grad()
            logps=model.forward(inputs)
            loss=criterion(logps,labels)
            loss.backward()
            optimizer.step()
        
            running_loss+=loss.item()
        
            if steps%print_every==0:
                validation_loss=0
                accuracy=0
                model.eval()
            
                with torch.no_grad():
                    for inputs,labels in validloader:
                        inputs,labels=inputs.to(device),labels.to(device)
                        logps=model.forward(inputs)
                        batch_loss=criterion(logps,labels)
                        validation_loss+=batch_loss.item()
                        #calculate accuracy
                        ps=torch.exp(logps)
                        top_p,top_class=ps.topk(1,dim=1)
                        equals=top_class==labels.view(*top_class.shape)
                        accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader)*100:.3f}%")
                running_loss = 0
                model.train()                                          

model.class_to_idx = train_data.class_to_idx

#creating checkpoint dictionary
checkpoint = {'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'class_to_idx': train_data.class_to_idx,
              'opt_state': optimizer.state_dict,
              #'num_epochs': epochs
             }

torch.save(checkpoint, save_dir)   
correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for image,label in testloader:
        images, labels = image.to(device),label.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total+=labels.size(0)
print('Accuracy on test images : %d%%' % (100 * correct / total))
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms,models
import matplotlib.pyplot as plt
from torch import nn,optim
import json
import argparse
from utility_fun import load_checkpoint,process_image,predict_im


parser = argparse.ArgumentParser(description='Time to test')
parser.add_argument('image_loc',action='store',help='Location of the image to be tested')
parser.add_argument('checkpoint',action='store',help='Location of tranined model/checkpoint')
parser.add_argument('--top_k',action='store',dest='top_k',help='Extract Top K probabilities',default='5')
parser.add_argument('--category_names',action='store',dest='cat_names',help='Initialization of category names',default='cat_to_name.json')
parser.add_argument('--gpu',action='store',dest='gpu',help='Device on GPU?',default='off')

args = parser.parse_args()
image_name=args.image_loc
checkpoint=args.checkpoint
top_k=int(args.top_k)
cat_names=args.cat_names
gpu=args.gpu

if gpu=='on':
    device='cuda'
else:
    device='cpu'


with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

#loading checkpoint and assign to a model
model=load_checkpoint(checkpoint,device)

#pre process the image
img_test =  process_image(image_name)
top_ps,final_indices=predict_im(image_name,model,top_k)

#mapping top indices to actual flower names
y=[cat_to_name[x] for x in final_indices]

df = pd.DataFrame(list(zip(y, top_ps)), columns =['Flower Name', 'Probability']) 
print(df)
print('The flower is :',df.iloc[0,0],'and the probablity is = ',df.iloc[0,1])


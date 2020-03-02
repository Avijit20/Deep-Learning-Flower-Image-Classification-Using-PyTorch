import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets,transforms,models
import matplotlib.pyplot as plt
#from train import get_pretrained_model
from torch import nn,optim
from PIL import Image
#%matplotlib inline

def load_dataset(data_direc):
    data_dir = data_direc
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms=transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    train_data=datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_data=datasets.ImageFolder(valid_dir,transform=valid_transforms)
    test_data=datasets.ImageFolder(test_dir,transform=test_transforms)


    trainloader =torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    validloader =torch.utils.data.DataLoader(valid_data,batch_size=64)
    testloader =torch.utils.data.DataLoader(test_data,batch_size=64)
    
    return testloader,validloader,testloader,train_data

def load_checkpoint(filepath,device):
   # Checkpoint for when using GPU
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    pretrained_model= 'vgg16' #get_pretrained_model()
    checkpoint = torch.load(filepath, map_location=map_location)
    model = getattr(models,pretrained_model)(pretrained=True)
    model.classifier=checkpoint['classifier']
    model.class_to_idx=checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    #device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im=Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    
    ## Transforming image for use with network
    trans_im = transform(im)
    
    # Converting to Numpy array 
    array_im_tfd = np.array(trans_im)
    
    return array_im_tfd

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    #image = image.numpy().transpose((1, 2, 0))
    image=image.transpose((1,2,0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict_im(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    k=topk
    test_img=process_image(image_path)
    test_img=torch.from_numpy(test_img)
    batch_img=torch.unsqueeze(test_img,0)
    outputs=model(batch_img)
    top_ps,top_indices = torch.topk(outputs,k)
    top_ps=torch.exp(top_ps)
    class_to_idx_inv={k:v for v,k in model.class_to_idx.items()}
    top_ps=top_ps.view(k).tolist()
    top_indices=top_indices.view(k).tolist()
    final_indices=[class_to_idx_inv[x] for x in top_indices]
    return top_ps,final_indices


    
  


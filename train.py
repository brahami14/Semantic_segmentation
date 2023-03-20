# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:29:38 2023

@author: SC43822
"""

import torch 
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import  torch.nn as nn 
import torch.optim as optim 
from model import UNET
from torchvision import transforms 
A=transforms
from utils import (load_checkpoint,save_checkpoint,get_loaders,check_accuracy)
#,save_predictions_as_imgs

import numpy as np
#Hyperparameters

LEARNING_RATE=1e-4
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
# DEVICE='cpu'
BATCH_SIZE=16
NUM_EPOCHS=2
NUM_WORKERS=4
IMAGE_HEIGHT=200
IMAGE_WIDTH=300
PIN_MEMORY=True
Load_Model=False
TRAIN_IMG_DIR="dataset/Train/road/images/"
TRAIN_MASK_DIR="dataset/Train/road/masks/"
VAL_IMG_DIR="dataset/Valid/road/images/"
VAL_MASK_DIR="dataset/Valid/road/masks/"

def train_fn(loader,val_loader,model,optimizer,loss_fn,scaler,epoch):
    loop=tqdm(total=len(loader))
    global_iter = 0
    check_iter = 7 #☺ evluate every n step 
   
    #training
    for batch_idx,(data,targets) in enumerate(loader):
        
     
        
        if global_iter % check_iter == 0 and epoch > 0:
                model.eval()
                hist_list = []
                val_loss_list = []
                
                with torch.no_grad():
                    #validation 
                    
                 for batch_idx_val,(data_val,targets_val) in enumerate(loader):
                        data_val=(data_val).to(device=DEVICE)
                        print("**predicts_label",torch.max(targets_val))
                        targets_val= torch.argmax(targets_val, dim=1).to(device=DEVICE)
                       
                        predicts_label=model(data_val)
                       
                        loss=loss_fn(predicts_label,targets_val)
                        
                        predicts_label=torch.argmax(predicts_label, dim=1).to(device=DEVICE)
                        
                        #print("**predicts_label",torch.max(predicts_label))
                        
                        
                        
                        val_loss_list.append(loss.detach().cpu().numpy())
                        
        
        
        # print('-----------batch_idx------------')
        # print('---------data-------------')
        data=data.to(device=DEVICE)
        
      
        targets= torch.argmax(targets, dim=1).to(device=DEVICE)
       
 
      
         # forward 
        with torch.cuda.amp.autocast():
             predictions=model(data)
             loss=loss_fn(predictions,targets)
        #backward
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.update()
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
        global_iter += 1
                         
    

def main():
    
    train_transfrom=transforms.Compose([
        A.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
        # A.RandomRotation(limit=35,p=0.1),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.1),
        #A.Totensor(),
        transforms.ToTensor()
        # A.Normalize(mean=[0.0,0.0,0.0],
        #             std=[1.0,1.0,1.0],
                 
                 
       
        ],
        )
    val_transforms=transforms.Compose([
        A.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
        # A.RandomRotation(limit=35,p=0.1),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.1),
        transforms.ToTensor(),
       
       
        ],
        )
    
    model =UNET(in_channels=3, out_channels=3).to(DEVICE)
    model.train()
    loss_fn=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE,eps=1e-4)
    train_loader,val_loader=get_loaders(TRAIN_IMG_DIR,TRAIN_MASK_DIR,VAL_IMG_DIR,VAL_MASK_DIR,BATCH_SIZE,train_transfrom,
                                       NUM_WORKERS,PIN_MEMORY)
    
    scaler=torch.cuda.amp.GradScaler()
    
    
    
    
  
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader,val_loader,model,optimizer,loss_fn,scaler,epoch)
        
       # save model 
     
        
        checkpoint={
                    "epoch":epoch,
                    'state_dict':model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    'loss':loss_fn,
                    }
        save_checkpoint(checkpoint, filename="trained_model/model_save.pt")
        
        # check accuracy
        check_accuracy(val_loader,model,device=DEVICE)
        
        #save_predictions_as_imgs(val_loader,model,folder=)
        
        
        
        
        
    
if __name__=="__main__":
    
    main()

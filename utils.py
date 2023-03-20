# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 15:19:01 2023

@author: SC43822
"""
import torch 
import torchvision 
from dataset import TrimbleDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename):
    print("=> saving checkpoint")
    torch.save(state,filename)
    
    
def load_checkpoint(checkpoint,model):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
    
def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transfrom,
        val_transform,
        num_workers,
        pin_memory=True):
    train_ds=TrimbleDataset(image_dir=train_dir,mask_dir=train_maskdir,transfrom=train_transfrom)
    
    train_loader=DataLoader(train_ds,batch_size=batch_size, num_workers=num_workers,pin_memory=pin_memory,
                            shuffle=False,)
    
        
    val_ds=TrimbleDataset(image_dir=val_dir,mask_dir=val_maskdir)
    
    val_loader=DataLoader(val_ds,batch_size=batch_size, num_workers=num_workers,pin_memory=pin_memory,
                            shuffle=False,)
    
    return train_loader, val_loader

def check_accuracy(loader,model,device="cuda"):
    num_correct=0
    num_pixels=0
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            y=y.to(device).unsqueeze(1)
            preds=torch.sigmoid(model(x))
            
            preds=(preds>0.5).float()
            num_correct+=(preds==y).sum()
            num_pixels+torch.numel(preds)
            
  
                
            
            
    
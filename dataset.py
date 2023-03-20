# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:29:36 2023

@author: SC43822
"""
import os

from PIL import Image 

from torch.utils.data import Dataset 
import numpy as np
import torchvision.transforms as T

class TrimbleDataset(Dataset):
    def __init__(self,image_dir,mask_dir,transfrom=None):
        self.image_dir=image_dir
        self.mask=mask_dir
        self.transform=transfrom
        self.images=os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        img_path=os.path.join(self.image_dir,self.images[index])
        mask_path=os.path.join(self.mask,self.images[index])
        
        image=Image.open(img_path).convert("RGB")
      
        mask=Image.open(mask_path).convert("L")
      
        image=image.resize((200,300))
        mask=mask.resize((200,300))
      
        # image = np.array(image)
        # mask = np.array(mask, dtype=np.float32)
        transform=T.ToTensor()
     
        if self.transform:
            image=transform(image)
            mask=transform(mask)
        else:
            image=transform(image)
            mask=transform(mask)
         
            
            
        
        return (image,mask)
        
        #mask[mask==255.0]=1.0
        
        # if self.transform is not None:
        #     augmentations=self.transform(image=image,mask=mask)
        #     image=augmentations[image]
        #     mask=augmentations[mask]
            
        
        
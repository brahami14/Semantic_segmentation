# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:57:49 2023

@author: SC43822
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 08:48:29 2023

@author: SC43822
"""
import os
from PIL import Image
import numpy as np 
path_images="dataset/Train/road/images/000000.png"
path_masks="dataset/Train/road/masks/000000.png"
import numpy as np
import torchvision.transforms as T
# j=0
# for i in range (1,53):
    
#     img_path="dataset/Train/road/roads/"+"Label_"+str(i)+".png"
#     image=Image.open(img_path).convert("RGB")
#     if (i<10 ):
#         img1_path="dataset/Train/road/masks1/"+"00000"+str(j)+".png"
#         image.save(img1_path)
#     if(i>10 and i <= 99) :
#         img1_path="dataset/Train/road/masks1/"+"0000"+str(j)+".png"
#         image.save(img1_path)
#     if(i>=100) :
#           img1_path="dataset/Train/road/masks1/"+"000"+str(j)+".png"
#           image.save(img1_path)
#     j=j+1
   
    


# j=81
# path=os.listdir(path_images)

# for i in enumerate(path[87:107]):
#     print(i[1])
    
#     img_path="dataset/Train/road/images1/"+str(i[1])
#     image=Image.open(img_path).convert("RGB")
#     if (j<10 ):
#         img1_path="dataset/Train/road/images1/"+"00000"+str(j)+".png"
#         image.save(img1_path)
#     if(j>10 and j <= 99) :
#         img1_path="dataset/Train/road/images1/"+"0000"+str(j)+".png"
#         image.save(img1_path)
#     if(j>=100) :
#           img1_path="dataset/Train/road/images1/"+"000"+str(j)+".png"
#           image.save(img1_path)
#     j=j+1
   
# img_path="dataset/Train/road/images/"+str(i)+".jpeg"
# image=Image.open(img_path).convert("RGB")
# if (i<10 ):
    
#    img1_path="dataset/Train/road/images1/"+"00000"+str(j)+".png"
#    image.save(img1_path)

image=Image.open(path_images).convert("RGB")
print(image)

mask=Image.open(path_masks).convert("L")
image=image.resize((100,100))

mask=mask.resize((100,100))
print('image', image)
print("mask",mask)

transform=T.ToTensor()
if transform:
    image=transform(image)
    mask=transform(mask)
print("after tensor")
print('image*****', np.shape(image))
print("****mask",np.shape(mask))

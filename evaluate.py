import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.dice_score import multiclass_dice_coeff, dice_coeff
import torch.nn as nn
from utils.dice_score import dice_loss

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    print("num_val_batches***",num_val_batches )
    # iterate over the validation set
    criterion = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        #for batch in dataloader:
            image, mask_true = batch['image'], batch['mask']
           
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            print("*****size of image for evaluation**********", np.shape(image))
            print("*****size of mask_true for evaluation", np.shape(mask_true))
            
            
                   
            if net.n_classes == 1:
                    loss = criterion(mask_pred.squeeze(1), true_masks.float())
                    loss += dice_loss(F.sigmoid(mask_pred.squeeze(1)), mask_true.float(), multiclass=False)
            else:
                
                loss = criterion(mask_pred, mask_true)
                loss += dice_loss(
                            F.softmax(mask_pred, dim=1).float(),
                            F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
             
                                                       
                
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    precision=dice_score / max(num_val_batches, 1)
   
    return precision,loss

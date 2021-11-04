import torch, os, cv2
import torch.nn as nn
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataset import MoNuSeg
from skimage.measure import label, regionprops

def save_checkpoint(filename, state, epoch):
    if not os.path.exists(filename):
      os.makedirs(filename)
    filename = filename + f"/weights_ep{epoch}.pth.tar"
    torch.save(state, filename)
    print(f"=> Just saved checkpoint for epoch : {epoch}")
    
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    print("checkpoint loaded!")
    
def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True,        
):
    
    train_ds = MoNuSeg(
        image_dir = train_dir, 
        mask_dir = train_maskdir,
        transform=train_transform
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,            
    )
    
    val_ds = MoNuSeg(
        image_dir = val_dir, 
        mask_dir = val_maskdir,   
        transform=val_transform
    )
     
    val_loader = DataLoader(
        val_ds,
        batch_size = 1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,            
    )    
    
    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou = 0; f1 = 0; PRECISION = 0; RECALL = 0; accuracy = 0; dice_score = 0
    model.eval()
    epsilon = 1e-7
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds, _ = model(x)
            preds = torch.sigmoid(preds)
            preds = (preds>0.5).float()

            tp = (y * preds).sum().to(torch.float32)
            tn = ((1 - y) * (1 - preds)).sum().to(torch.float32)
            fp = ((1 - y) * preds).sum().to(torch.float32)
            fn = (y * (1 - preds)).sum().to(torch.float32)
            
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            PRECISION += precision
            RECALL += recall
            
            f1 += 2* (precision*recall) / (precision + recall + epsilon)
            accuracy += (tp + tn)/(tp + fp + tn + fn)

            intersection = (preds*y).sum()
            iou += (intersection/(epsilon + (preds + y).sum() - intersection)) # iou = (P int Q)/(P + Q - (P int Q))
            dice_score += (2*(preds*y).sum()) / (epsilon + (preds + y).sum())           
    model.train()
    return iou/len(loader), f1/len(loader), PRECISION/len(loader), RECALL/len(loader), accuracy/len(loader), dice_score/len(loader)

def produce_output_masks(test_loader, model, path, device="cuda", BATCH_SIZE = 1, save = True):
    model.eval()
    if not os.path.exists(path):
      os.makedirs(path)
    k = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            mask, boundary = model(x)
            mask = torch.sigmoid(mask)
            boundary = torch.sigmoid(boundary)
            
            mask = (mask>0.5).float()
            boundary = (boundary>0.5).float()
            img = x.cpu().numpy().transpose(0, 2, 3, 1)
            gt = y.cpu().numpy().transpose(0, 2, 3, 1)
            mask = mask.cpu().numpy().transpose(0, 2, 3, 1)
            boundary = boundary.cpu().numpy().transpose(0, 2, 3, 1)
           
            for b in range(BATCH_SIZE):
              if not save:
                plt.figure(figsize = (16, 16))
                plt.subplot(141); plt.imshow(img[b])
                plt.subplot(142); plt.imshow(gt[b]*255, cmap = 'gray')
                plt.subplot(143); plt.imshow(mask[b]*255, cmap = 'gray')
                plt.subplot(144); plt.imshow(boundary[b]*255, cmap = 'gray')
              else:
                cv2.imwrite(os.path.join(path, f'{k}_img.png'), img[b]*255)
                cv2.imwrite(os.path.join(path, f'{k}_gt.png'), gt[b]*255)
                cv2.imwrite(os.path.join(path, f'{k}_pred.png'), mask[b]*255)
                cv2.imwrite(os.path.join(path, f'{k}_fore.png'), boundary[b]*255)
            k += 1         
    model.train()
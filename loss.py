import torch
import torch.nn as nn
import numpy as np
import cv2


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()
    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)             
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)    
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()      
        Tversky = TP / (TP + alpha*FP + beta*FN + 1e-7)          
        return 1 - Tversky
    
class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
    def forward(self, inputs, targets):
        # calculate edges
        pred = inputs
        x_size = targets.size()
        backup_targets = targets.clone().detach()
        im_arr = backup_targets.cpu().numpy().transpose((0,2,3,1))
        im_arr = (im_arr*255).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            a = im_arr[i]
            canny[i] = cv2.Canny(a,10,250)
            canny[i] = (canny[i]>0)
        canny_gt = torch.from_numpy(canny)#.cuda().float()   
        canny_gt = canny_gt.to('cuda').to(dtype=torch.float32) 
        
        loss = nn.BCEWithLogitsLoss()
        loss1 = loss(pred, canny_gt)
        loss = nn.MSELoss()
        loss2 = loss(pred, canny_gt)
        return loss1 + loss2

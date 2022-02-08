import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import UNet, attn_guided_global_brach

class stage1_enc_dec(nn.Module):
  def __init__(self):
    super(stage1_enc_dec, self).__init__()
    self.local_net = UNet(3, 1) # seond parameter is not needed to be defined
    self.global_net = attn_guided_global_brach()
    
    self.conv1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, padding = 1, stride = 1), ### ks changed from 1 to 3
                              nn.BatchNorm2d(32),
                              nn.ReLU(inplace=True))
    
    self.conv2 = nn.Conv2d(32, 1, kernel_size = 1, padding = 0, stride = 1)                              
    
  def forward(self, x):
    x_ = torch.add(self.local_net(x), self.global_net(x))
    x1 = self.conv1(x_)
    x2 = self.conv2(x1)
    return x1, x2 # x1 : 32 channels, x2 : 1 channel (segmentation map)
  
class stage1_enc_dec(nn.Module):
  def __init__(self):
    super(stage1_enc_dec, self).__init__()
    self.local_net = UNet(3, 1) # seond parameter is not needed to be defined
    self.global_net = attn_guided_global_brach()
    
    self.conv1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 1, padding = 0, stride = 1),
                              nn.BatchNorm2d(32),
                              nn.ReLU())
    
    
  def forward(self, x):
    x = torch.add(self.local_net(x), self.global_net(x))
    x1 = self.conv1(x)
    return x1 

class stage2_enc_dec(nn.Module):
    # this stage contains both the mask and boundary generators
    def __init__(self):
        super(stage2_enc_dec, self).__init__()
        self.conv1_mask = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3, padding = 1, stride = 1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        self.conv2_mask = nn.Sequential(nn.Conv2d(64, 128, kernel_size = 3, padding = 1, stride = 1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.Conv2d(128, 128, kernel_size = 3, padding = 1, stride = 1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.maxpool_mask = nn.MaxPool2d(kernel_size = 2, stride = 2) 
        self.upsample_mask = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.conv1_boundary = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3, padding = 1, stride = 1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        self.conv2_boundary = nn.Sequential(nn.Conv2d(64, 128, kernel_size = 3, padding = 1, stride = 1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.Conv2d(128, 128, kernel_size = 3, padding = 1, stride = 1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.maxpool_boundary = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.upsample_boundary = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.conv_final_mask = nn.Sequential(nn.Conv2d(128, 32, kernel_size = 3, padding = 1, stride = 1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 1, kernel_size = 1, padding = 0, stride = 1))
        self.conv_final_boundary = nn.Sequential(nn.Conv2d(128, 32, kernel_size = 3, padding = 1, stride = 1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 1, kernel_size = 1, padding = 0, stride = 1))
    def forward(self, x):
        xm1 = self.conv1_mask(x)
        xb1 = self.conv1_boundary(x)
        xm = self.maxpool_mask(xm1)
        xb = self.maxpool_boundary(xb1)
        xm = self.conv2_mask(xm)
        xb = self.conv2_boundary(xb)
        xm = xm + xb
        xm = self.upsample_mask(xm)
        xb = self.upsample_boundary(xb)
        xm = torch.cat((xm, xm1), dim=1)
        
        xb = torch.cat((xb, xb1), dim=1) 

        mask = self.conv_final_mask(xm)
        boundary = self.conv_final_boundary(xb)
        return mask, boundary
        
class Final_Model(nn.Module):
    def __init__(self):
        super(Final_Model, self).__init__()
        self.GearSecond = stage2_enc_dec()
        self.GearFirst = stage1_enc_dec()
        # self.freeze_G1()

    def freeze_G1(self):
        for param in self.GearFirst.parameters():
          param.requires_grad = False

    def forward(self, input):
        x, _ = self.GearFirst(input) # 32 ch, 1 ch (input)
        # x = torch.cat((x, input), dim = 1)
        mask, boundary = self.GearSecond(x)
        return mask, boundary # mask, boundary predictions
    
    

import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import reduce
import pdb
from torch.autograd import Variable
import torchvision
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SAM_Module(nn.Module):
    """ Spatial attention module"""
    def __init__(self, in_dim):
        super(SAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, padding = 0)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, padding = 0)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, padding = 0)
        self.pool_sizes = [1, 3, 7, 11]
        self.pool1 = nn.AdaptiveAvgPool2d((self.pool_sizes[0], self.pool_sizes[0]))
        self.pool2 = nn.AdaptiveAvgPool2d((self.pool_sizes[1], self.pool_sizes[1]))
        self.pool3 = nn.AdaptiveAvgPool2d((self.pool_sizes[2], self.pool_sizes[2]))
        self.pool4 = nn.AdaptiveAvgPool2d((self.pool_sizes[3], self.pool_sizes[3]))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        residual = x
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1) # shape : (BS, HW, C//8)
        proj_key = self.key_conv(x) # .view(m_batchsize, -1, width*height)
        proj_value = self.value_conv(x)
        x = proj_key
        flat1 = self.pool1(x).view(m_batchsize, C, -1)
        flat2 = self.pool2(x).view(m_batchsize, C, -1)
        flat3 = self.pool3(x).view(m_batchsize, C, -1)
        flat4 = self.pool4(x).view(m_batchsize, C, -1)
        x = torch.cat((flat1, flat2, flat3, flat4), dim = 2) # shape : (BS, C//8, S = 110)

        energy = torch.bmm(proj_query, x) 
        attention = self.softmax(energy)# shape : (BS, N, S), N = HW

        x = proj_value
        flat1 = self.pool1(x).view(m_batchsize, C, -1)
        flat2 = self.pool2(x).view(m_batchsize, C, -1)
        flat3 = self.pool3(x).view(m_batchsize, C, -1)
        flat4 = self.pool4(x).view(m_batchsize, C, -1)
        x = torch.cat((flat1, flat2, flat3, flat4), dim = 2) # shape : (BS, C//8, S = 110)
        x = x.permute(0, 2, 1)
        out = torch.bmm(attention, x) # shape : (BS, N, C//8)

        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + residual
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.pool_sizes = [1, 3, 7, 11]
        self.pool1 = nn.AdaptiveAvgPool2d((self.pool_sizes[0], self.pool_sizes[0]))
        self.pool2 = nn.AdaptiveAvgPool2d((self.pool_sizes[1], self.pool_sizes[1]))
        self.pool3 = nn.AdaptiveAvgPool2d((self.pool_sizes[2], self.pool_sizes[2]))
        self.pool4 = nn.AdaptiveAvgPool2d((self.pool_sizes[3], self.pool_sizes[3]))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        residual = x
        proj_query = x
        x = proj_query
        flat1 = self.pool1(x).view(m_batchsize, C, -1)
        flat2 = self.pool2(x).view(m_batchsize, C, -1)
        flat3 = self.pool3(x).view(m_batchsize, C, -1)
        flat4 = self.pool4(x).view(m_batchsize, C, -1)
        x1 = torch.cat((flat1, flat2, flat3, flat4), dim = 2) # shape : (BS, C, S = 110)

        proj_key = residual#.view(m_batchsize, C, -1).permute(0, 2, 1)
        x = proj_key
        flat1 = self.pool1(x).view(m_batchsize, C, -1)
        flat2 = self.pool2(x).view(m_batchsize, C, -1)
        flat3 = self.pool3(x).view(m_batchsize, C, -1)
        flat4 = self.pool4(x).view(m_batchsize, C, -1)
        x2 = torch.cat((flat1, flat2, flat3, flat4), dim = 2).permute(0, 2, 1) # shape : (BS, S = 110, C)

       
        energy = torch.bmm(x1, x2)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = residual.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + residual
        return out
    
class SAM_CAM_Layer(nn.Module):
    def __init__(self, in_ch, use_pam = True):
        super(SAM_CAM_Layer, self).__init__()
        
        self.attn = nn.Sequential(
            # nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            # nn.BatchNorm2d(in_ch),
            # nn.ReLU(),
            SAM_Module(in_ch) if use_pam else CAM_Module(in_ch),
			      nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.attn(x)

class Unet_encoder(nn.Module):
    def __init__(self):
        super(Unet_encoder, self).__init__()
        self.conv0_enc = nn.Sequential(nn.Conv2d(3, 32, 3, stride = 2, padding = 1, bias = False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(32, 32, 3, stride = 1, padding = 1, bias = False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace = True))
        self.conv1_enc = nn.Sequential(nn.Conv2d(32, 64, 3, stride = 1, padding = 1, bias = False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(64, 64, 3, stride = 1, padding = 1, bias = False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace = True))
        self.pool1 = nn.Conv2d(64, 64, 3, stride = 2, padding = 1)
        self.conv2_enc = nn.Sequential(nn.Conv2d(64, 128, 3, stride = 1, padding = 1, bias = False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(128, 128, 3, stride = 1, padding = 1, bias = False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(inplace = True))
        self.pool2 = nn.Conv2d(128, 128, 3, stride = 2, padding = 1)
        self.conv3_enc = nn.Sequential(nn.Conv2d(128, 256, 3, stride = 1, padding = 1, bias = False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(256, 256, 3, stride = 1, padding = 1, bias = False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace = True))
        self.pool3 = nn.Conv2d(256, 256, 3, stride = 2, padding = 1)
        self.conv4_enc = nn.Sequential(nn.Conv2d(256, 512, 3, stride = 1, padding = 1, bias = False),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(512, 512, 3, stride = 1, padding = 1, bias = False),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(inplace = True))
        
    def forward(self, inp):
        x = self.conv0_enc(inp) # 32x256x256
        x256 = self.conv1_enc(x) # 64x256x256
        x = self.pool1(x256)

        x128 = self.conv2_enc(x) # 128x128x128
        x = self.pool2(x128)

        x64 = self.conv3_enc(x)
        x = self.pool3(x64)

        x32 = self.conv4_enc(x)   
        return x256, x128, x64, x32
    
class attn_guided_global_brach(nn.Module):
    def __init__(self):
        super(attn_guided_global_brach, self).__init__()
        self.encoder = Unet_encoder()

        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.PAM1 = PAM_CAM_Layer(64, True) 
        self.CAM1 = PAM_CAM_Layer(64, False)

        self.PAM2 = PAM_CAM_Layer(128, True) 
        self.CAM2 = PAM_CAM_Layer(128, False)

        self.PAM3 = PAM_CAM_Layer(256, True) 
        self.CAM3 = PAM_CAM_Layer(256, False) 

        self.PAM4 = PAM_CAM_Layer(512, True) 
        self.CAM4 = PAM_CAM_Layer(512, False) 

        self.conv1_enc = nn.Sequential(nn.Conv2d(64 + 128, 128, kernel_size = 1), 
                                      nn.BatchNorm2d(128),
                                      nn.ReLU())
        self.conv2_enc = nn.Sequential(nn.Conv2d(128 + 256, 256, kernel_size = 1), 
                                      nn.BatchNorm2d(256),
                                      nn.ReLU())
        self.conv3_enc = nn.Sequential(nn.Conv2d(256 + 512, 512, kernel_size = 1), 
                                      nn.BatchNorm2d(512),
                                      nn.ReLU())
        
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size = 1), 
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size = 1), 
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size = 3, padding = 1), 
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, kernel_size = 3, padding = 1), 
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        
        self.conv_final = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                                      nn.Conv2d(32, 32, kernel_size = 3, padding = 1, stride = 1), 
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 32, kernel_size = 3, padding = 1, stride = 1), 
                                      nn.BatchNorm2d(32),
                                      nn.ReLU())
                                      
        
    def forward(self, x):
        layer = self.encoder(x)
        layer1 = layer[0] # 64 x 256 x 256 
        layer2 = layer[1] # 128 x 128 x 128
        # layer3 = layer[2] # 256 x 64 x 64
        # layer4 = layer[3] # 512 x 32 x 32

        pam1 = self.PAM1(layer1)
        cam1 = self.CAM1(layer1)
        refined1 = torch.add(pam1, cam1)
        refined1 *= layer1

#         layer1_downsampled = self.maxpool(layer1)
#         layer2 = torch.cat((layer2, layer1_downsampled), dim = 1)
#         layer2 = self.conv1_enc(layer2)
        pam2 = self.PAM2(layer2)
        cam2 = self.CAM2(layer2)
        refined2 = torch.add(pam2, cam2)
        refined2 *= layer2
        
        ## decoder starts
        down2 = self.conv3(refined2) # 64, 256, 256, does an upsampling

        down1 = torch.cat((down2, refined1), dim=1) # 128, 256, 256
        down_final = self.conv4(down1) #64, 256, 256
        down_final += layer1 # skip connection from layer1
        down_final = self.conv_final(down_final) # 32, 512, 512 

        return down_final
    
# UNet building blocks have been reused from : https://github.com/cosmic-cortex/pytorch-UNet/tree/master/unet

class First2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class Encoder2D(nn.Module):
    def __init__(
            self, in_channels, middle_channels, out_channels,
            dropout=False, downsample_kernel=2
    ):
        super(Encoder2D, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=downsample_kernel),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Center2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center2D, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.center = nn.Sequential(*layers)

    def forward(self, x):
        return self.center(x)


class Decoder2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Last2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, softmax=False):
        super(Last2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels)
#             nn.Dropout2d(0.3)
            # nn.ReLU(inplace=True),
            # nn.Conv2d(middle_channels, out_channels, kernel_size=1),
            # nn.Softmax(dim=1)
        ]

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, conv_depths = (32, 64, 128, 256, 512)):#conv_depths=(64, 128, 256, 512, 1024)):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'

        super(UNet, self).__init__()

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First2D(in_channels, conv_depths[0], conv_depths[0]))
        encoder_layers.extend([Encoder2D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1])
                               for i in range(len(conv_depths)-2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder2D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i])
                               for i in reversed(range(len(conv_depths)-2))])
        decoder_layers.append(Last2D(conv_depths[1], conv_depths[0], out_channels))

        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center2D(conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2])
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x, return_all=False):
        x_enc = [x]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))

        x_dec = [self.center(x_enc[-1])]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1-dec_layer_idx]
            x_cat = torch.cat(
                [pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))

        if not return_all:
            return x_dec[-1]
        else:
            return x_enc + x_dec


def pad_to_shape(this, shp):
    """
    Pads this image with zeroes to shp.
    Args:
        this: image tensor to pad
        shp: desired output shape
    Returns:
        Zero-padded tensor of shape shp.
    """
    if len(shp) == 4:
        pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    elif len(shp) == 5:
        pad = (0, shp[4] - this.shape[4], 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    return F.pad(this, pad)

### 任意大小。x编码.nomask
from tkinter import X
import logging

import torch
import torch.nn as nn


# from mamba.models.layers.combinations import *
# from mamba.models.layers.brt_modules import BlockRecurrentAttention
# from mamba.models.layers.network_swinir import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

import logging
from torch.cuda.amp import autocast
import torch
import torch.nn as nn
from mamba_ssm import Mamba

from einops import rearrange


def continues_scan(input):
    # t,h,w = input.shape
    # input = input.pad((0,1,0,1,0,1), mode='constant')
    # t_,h_,w_ = input.shape
    #print(input.shape)
    input_rev = input.flip(-1)#双向
    #print(input_rev.shape)
    input_ = input.clone()
    input_[:,:,:,1::2,:] = 0
    input_rev[:,:,:,0::2,:] = 0
    input_ = torch.add(input_, input_rev)
    # input_rev = input.flip(-2).contiguous()
    # input_rev[:,:,1:2] = 0
    # input[:,:,0:1] = 0
    # input = input+input_rev
    input_rev_ = input_.flip(-2).flip(-1)
    input_rev_[:,:,0::2,:,:] = 0
    input_[:,:,1::2,:,:] = 0
    # output = input+input_rev
    output = torch.add(input_, input_rev_)
    return output

def rev_continues_scan(input):
    input_rev = input.flip(-1).flip(-2)
    input_ = input.clone()
    input_rev[:,:,0::2,:,:] = 0
    input_[:,:,1::2,:,:] = 0
    input_ = torch.add(input_, input_rev)
    input_rev_ = input_.flip(-1)
    input_[:,:,:,1::2,:] = 0
    input_rev_[:,:,:,0::2,:] = 0
    output = torch.add(input_, input_rev_)
    return output

class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
        ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False,SSAS=None):
        super().__init__()
        #print(f"MambaLayer: dim: {dim}")
        self.SSAS = SSAS
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=32, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand    # Block expansion factor
        )
        self.mamba_b=Mamba(
                d_model=32, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand    # Block expansion factor
        )
        self.channel_token = channel_token ## whether to use channel as tokens

    def forward_patch_token(self, x, r = False):
        B, C, H, W = x.shape
        x=x.reshape(B,C//1,1,H,W)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> ' + self.SSAS, b=B, c=C, t=T, h=H, w=W)
        #print(x.shape)
        x = continues_scan(x)
        #print(x.shape)
        B, d_model = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)
        #print(x_flat.shape)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        x_mamba_b = self.mamba_b(x_norm.flip(1)).flip(1)
        x_mamba = x_mamba + x_mamba_b
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)
        out = rev_continues_scan(out).contiguous()
        out = rearrange(out , self.SSAS + ' -> b c t h w', b=B, c=C, t=T, h=H, w=W)
        out=out.reshape(B,-1,H,W)
        return out

    def forward_channel_token(self, x, r= False):
        B, n_tokens = x.shape[:2]
        
        img_dims = x.shape[2:]
        x_flat = x.flatten(2) 
        x_flat=x_flat.permute(0,2,1)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm).permute(0,2,1)
        out = x_mamba.reshape(B, n_tokens, *img_dims)

        return out

    @autocast(enabled=False)
    def forward(self, x,r=False):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x, r=r)
        else:
            out = self.forward_patch_token(x, r=r)

        return out

class cnnmodule(nn.Module):
    def __init__(self, channels):
        super(cnnmodule, self).__init__()

        self.dwc = nn.Sequential(nn.Conv2d(channels, channels, groups=channels,kernel_size=1),
                                   nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv3d(channels, channels,kernel_size=(1,1,3),padding=(0,0,1)),
                                   nn.ReLU())
        self.pwc= nn.Sequential(nn.Conv2d(2*channels, channels, kernel_size=1),
                                   nn.ReLU())
    def forward(self, x):
        #print(self.dwc(x).shape,self.conv1(x.unsqueeze(2)).squeeze(2).shape)
        x=self.pwc(torch.cat((self.dwc(x),self.conv1(x.unsqueeze(2)).squeeze(2)),dim=1))+x
        return x

class mymamba(nn.Module):
    def __init__(self):
        super(mymamba, self).__init__()
        self.mamba = MambaLayer(dim = 100,  SSAS='b c t h w')
        self.mamba1 = MambaLayer(dim = 100,  SSAS='b c h t w')
        #self.conv=cnnmodule(channels=100)
        self.avgpool=nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp=nn.Sequential(nn.Linear(100, 50),
                                nn.ReLU(),
                                nn.Linear(50, 100))
        self.conv=residual_block(in_channel=100,out_channel=100)
        self.spa_mamba = SpaMamba(100,use_residual=True,group_num=4)
        self.spe_mamba = SpeMamba(100,token_num=25,use_residual=True,group_num=4)

    def forward(self, x):  # x
        # print(x.shape)
        #print(x.shape)    
        x=self.conv(x.unsqueeze(2)).squeeze(2)
        x=self.mamba(x)+self.spe_mamba(x) 
        return x

def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer

class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): #(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        x2 = self.conv3(x1) #(1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x+x2, inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        return out


class SpeMamba(nn.Module):
    def __init__(self,channels, token_num=8, use_residual=True, group_num=4):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels/token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba( # This module uses roughly 3 * expand * d_model^2 parameters
                            d_model=self.group_channel_num,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=2,  # Block expansion factor
                            )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self,x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self,x):
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        if self.use_residual:
            return x + x_proj
        else:
            return x_proj


class SpaMamba(nn.Module):
    def __init__(self,channels,use_residual=True,group_num=4,use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
                           d_model=channels,  # Model dimension d_model
                           d_state=16,  # SSM state expansion factor
                           d_conv=4,  # Local convolution width
                           expand=2,  # Block expansion factor
                           )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self,x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B,H,W,C = x_re.shape
        x_flat = x_re.view(1,-1, C)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon
            
class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()
        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)
    




if __name__ == '__main__':
    model = mymamba().to("cuda:1")
    x = torch.randn(20,100,9,9).to("cuda:1")
    y = model(x)
    print(y.shape)

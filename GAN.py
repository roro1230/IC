from flask import Flask, request, jsonify, send_file
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm.notebook import tqdm
import kornia.color as color
from datetime import datetime
from pathlib import Path
import os
from pathlib import Path
from torchvision.transforms import ToPILImage
import os
from DCGAN import DCGenerator, Preprocessor
from CNN import CNNColorization

# Dropout layer that works even in the evaluation mode
class DropoutAlways(nn.Dropout2d):
    def forward(self, x):
        return F.dropout2d(x, self.p, training=True)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, padding_mode='reflect', bias=False if normalize else True),
            nn.InstanceNorm2d(out_channels, affine=True) if normalize else nn.Identity(), # Note that nn.Identity() is just a placeholder layer that returns its input.
            nn.LeakyReLU(0.2),
        )       

    def forward(self, x):
        return self.block(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=False, activation='relu'):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False if normalize else True),
            nn.InstanceNorm2d(out_channels, affine=True) if normalize else nn.Identity(),
            DropoutAlways(p=0.5) if dropout else nn.Identity(),
            nn.ReLU() if activation == 'relu' else nn.Tanh(),
        )
        
    def forward(self, x):
        return self.block(x)

class BlockD(nn.Module):
    def __init__(self, in_channels, out_channels, stride, normalize=True):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias = False if normalize else True, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels) if normalize else nn.Identity(),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.model(x)
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder1 = DownBlock(1, 64, normalize=False) # 256x256 -> 128x128
        self.encoder2 = DownBlock(64, 128) # 128x128 -> 64x64
        self.encoder3 = DownBlock(128, 256) # 64x64 -> 32x32
        self.encoder4 = DownBlock(256, 512) # 32x32 -> 16x16
        self.encoder5 = DownBlock(512, 512) # 16x16 -> 8x8
        self.encoder6 = DownBlock(512, 512) # 8x8 -> 4x4
        self.encoder7 = DownBlock(512, 512) # 4x4 -> 2x2
        self.encoder8 = DownBlock(512, 512, normalize=False) # 2x2 -> 1x1

        # Decoder
        self.decoder1 = UpBlock(512, 512, dropout=True) # 1x1 -> 2x2
        self.decoder2 = UpBlock(512*2, 512, dropout=True) # 2x2 -> 4x4
        self.decoder3 = UpBlock(512*2, 512, dropout=True) # 4x4 -> 8x8
        self.decoder4 = UpBlock(512*2, 512) # 8x8 -> 16x16
        self.decoder5 = UpBlock(512*2, 256) # 16x16 -> 32x32
        self.decoder6 = UpBlock(256*2, 128) # 32x32 -> 64x64
        self.decoder7 = UpBlock(128*2, 64) # 64x64 -> 128x128
        self.decoder8 = UpBlock(64*2, 2, normalize=False, activation='tanh') # 128x128 -> 256x256

    def forward(self, x):
        # Encoder
        ch256_down = x
        ch128_down = self.encoder1(ch256_down)
        ch64_down = self.encoder2(ch128_down)
        ch32_down = self.encoder3(ch64_down)
        ch16_down = self.encoder4(ch32_down)
        ch8_down = self.encoder5(ch16_down)
        ch4_down = self.encoder6(ch8_down)
        ch2_down = self.encoder7(ch4_down)
        ch1 = self.encoder8(ch2_down)

        # Decoder
        ch2_up = self.decoder1(ch1)
        ch2 = torch.cat([ch2_up, ch2_down], dim=1)
        ch4_up = self.decoder2(ch2)
        ch4 = torch.cat([ch4_up, ch4_down], dim=1)
        ch8_up = self.decoder3(ch4)
        ch8 = torch.cat([ch8_up, ch8_down], dim=1)
        ch16_up = self.decoder4(ch8)
        ch16 = torch.cat([ch16_up, ch16_down], dim=1)
        ch32_up = self.decoder5(ch16)
        ch32 = torch.cat([ch32_up, ch32_down], dim=1)
        ch64_up = self.decoder6(ch32)
        ch64 = torch.cat([ch64_up, ch64_down], dim=1)
        ch128_up = self.decoder7(ch64)
        ch128 = torch.cat([ch128_up, ch128_down], dim=1)
        ch256_up = self.decoder8(ch128)
        
        return ch256_up
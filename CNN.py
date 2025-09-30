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
import numpy as np


class CNNColorization(nn.Module):
    def __init__(self):
        super(CNNColorization, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1)  # Input: 1 channel (grayscale)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)

        # Concatenation
        self.conv8 = nn.Conv2d(64 + 1, 64, kernel_size=3, padding=1, stride=1)
        self.bn6 = nn.BatchNorm2d(64)

        # Output layer
        self.conv9 = nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1)
        self.conv10 = nn.Conv2d(32, 2, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        # Encoder
        x1 = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x1))
        x2 = self.bn1(x2)
        x2 = self.pool1(x2)

        x3 = F.leaky_relu(self.conv3(x2))
        x3 = self.bn2(x3)
        x3 = self.pool2(x3)

        x4 = F.leaky_relu(self.conv4(x3))
        x4 = self.bn3(x4)
        x5 = F.leaky_relu(self.conv5(x4))
        x5 = self.bn4(x5)

        # Decoder
        x6 = self.up1(x5)
        x6 = F.leaky_relu(self.conv6(x6))
        x6 = self.bn5(x6)

        x7 = self.up2(x6)
        x7 = F.leaky_relu(self.conv7(x7))

        # Resize input tensor x to match the spatial size of x7
        if x7.shape[-2:] != x.shape[-2:]:
            x = F.interpolate(x, size=x7.shape[-2:], mode='bilinear', align_corners=False)

        # Concatenation
        concat = torch.cat([x7, x], dim=1)
        x8 = F.leaky_relu(self.conv8(concat))
        x8 = self.bn6(x8)

        # Output
        x9 = F.leaky_relu(self.conv9(x8))
        out = torch.tanh(self.conv10(x9))

        return out


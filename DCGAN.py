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

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
IMAGE_SIZE = 256
NOISE_DIM = 100

class DCGenerator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM):
        super(DCGenerator, self).__init__()
        self.noise_dim = noise_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),  # 512 x 4 x 4
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),   # 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),   # 32 x 64 x 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),   # 16 x 128 x 128
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 2, kernel_size=4, stride=2, padding=1, bias=False),   # 2 x 256 x 256
            nn.Tanh()
        )

    def forward(self, noise):
        return self.model(noise)
    
    
class Preprocessor(nn.Module):
    def __init__(self, input_channels=1, noise_dim=NOISE_DIM):
        super(Preprocessor, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=4, stride=2, padding=1),  # 128 x 128 x 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1),  # 16 x 16 x 16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16, noise_dim, kernel_size=1)       # NOISE_DIM x 1 x 1
        )

    def forward(self, x):
        return self.process(x)


class DCDiscriminator(nn.Module):
    def __init__(self):
        super(DCDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 64 x 128 x 128
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),  # 512 x 16 x 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, bias=False),  # 1 x 16 x 16
            nn.Sigmoid()
        )

    def forward(self, Lab):
        return self.model(Lab)
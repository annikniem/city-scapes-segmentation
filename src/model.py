"""
FCN-8 Model for Semantic Segmentation

This file contains the Fully Convolutional Neural Network Architecture (FCN-8).
The model is used for semantic segmentation tasks. It makes use of convolutional layers
for feature detection, max pooling for down sampling, ReLU for non-linear activations,
transposed convolutions for up-sampling and skip-connections for improved segmentation results.
"""

# %% Imports
import torch.nn as nn

# %% Model architecture
class FCN_8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Defining the down sampling layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=100)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # Defining the bottle neck
        self.bneck1 = nn.Conv2d(512, 4096, kernel_size=7, padding=0)
        self.bneck2 = nn.Conv2d(4096, 4096, kernel_size=1, padding=0)
        
        # Defining the activation function, drop out and pooling layers
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.drop = nn.Dropout2d(p=0.5)
        
        # Defining the up sampling layers
        self.score1 = nn.Conv2d(4096, out_channels, kernel_size=1, padding=0)
        self.upscore1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2)
     
        self.score2 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.upscore2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2)

        self.score3 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.upscore3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=16, stride=8)
        
    def forward(self, x):
        # Encoder
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1))
        x3 = self.pool(x2) # 64 channels
        
        x4 = self.act(self.conv3(x3))
        x5 = self.act(self.conv4(x4))
        x6 = self.pool(x5) # 128 channels
        
        x7 = self.act(self.conv5(x6))
        x8 = self.act(self.conv6(x7))
        x9 = self.act(self.conv6(x8))
        x10 = self.pool(x9) # 256 channels
        
        x11 = self.act(self.conv7(x10))
        x12 = self.act(self.conv8(x11))
        x13 = self.act(self.conv8(x12))
        x14 = self.pool(x13) # 512 channels
        
        x15 = self.act(self.conv8(x14))
        x16 = self.act(self.conv8(x15))
        x17 = self.act(self.conv8(x16))
        x18 = self.pool(x17) # 512 channels
        
        # Bottle neck
        x19 = self.act(self.bneck1(x18))
        x20 = self.drop(x19)
        x21 = self.act(self.bneck2(x20)) # 4096 channels
        x22 = self.drop(x21)
        
        # Decoder
        x23 = self.score1(x22) # 34 channels
        x24 = self.upscore1(x23) # 34 channels
        x25 = self.score2(x14) # 512 channels?
        
        x26 = x25[:, :, 5:5 + x24.size()[2], 5:5 + x24.size()[3]]                          
        x27 = x24 + x26  # Skip connection
        
        x28 = self.upscore2(x27)                                                          
        x29 = self.score3(x10)
        
        x30 = x29[:, :, 9:9 + x28.size()[2], 9:9 + x28.size()[3]]                         
        x31 = x28 + x30  # Skip connection                                                    
        
        x32 = self.upscore3(x31)                                                        
        x33 = x32[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]]
        
        return x33

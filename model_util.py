from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_size, stride, padding):
        super(ConvolutionBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.layers(x)
    
class Convolution2dTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Convolution2dTransposeBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.layers(x)
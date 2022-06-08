import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from utils import modelSummary, train_evaluate, plot_training_results
from model_util import ConvolutionBlock, Convolution2dTransposeBlock
import os

class Discriminator(nn.Module):
    def __init__(self, input_channels, features_dim):
        super(Discriminator, self).__init__()
        
        self.conv1 =  nn.Conv2d(input_channels, features_dim, kernel_size=4, stride=2, padding=1)
        
        self.conv2 = ConvolutionBlock(features_dim, features_dim * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = ConvolutionBlock(features_dim*2, features_dim * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = ConvolutionBlock(features_dim*4, features_dim * 8, kernel_size=4, stride=2, padding=1)
        
        self.conv5 = nn.Conv2d(features_dim*8, 1, kernel_size=4, stride=1, padding=0)
        

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.sigmoid(x)
        return x

class Generator(nn.Module):
    def __init__(self, noise_channels, input_channels, features_gen_dim):
        super(Generator, self).__init__()
        self.conv2d1 = Convolution2dTransposeBlock(noise_channels, features_gen_dim * 16, kernel_size=4, stride=1, padding=0)
        self.conv2d2 = Convolution2dTransposeBlock(features_gen_dim * 16, features_gen_dim * 8, kernel_size=4, stride=2, padding=1)
        self.conv2d3 = Convolution2dTransposeBlock(features_gen_dim * 8, features_gen_dim * 4, kernel_size=4, stride=2, padding=1)
        self.conv2d4 = Convolution2dTransposeBlock(features_gen_dim * 4, features_gen_dim * 2, kernel_size=4, stride=2, padding=1)
        
        self.conv2d5 = nn.ConvTranspose2d(features_gen_dim * 2, input_channels, kernel_size=4, stride=2, padding=1)
        
        

    def forward(self, x):
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x = self.conv2d4(x)
        x = torch.tanh(self.conv2d5(x))
        return x


if __name__ == '__main__':
    LEARNING_RATE = 2e-4
    FEATURES_DISC_DIM = 64
    FEATURES_GEN_DIM = 64
    FEATURES_NOISE_DIM = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    discriminator = Discriminator(input_channels=3, features_dim=FEATURES_DISC_DIM).to(device)
    generator = Generator(noise_channels=FEATURES_NOISE_CHANNELS, input_channels=3, features_gen_dim=FEATURES_GEN_DIM).to(device)
    
    modelSummary(discriminator)
    modelSummary(generator)
    
    training_params = {
    'num_epochs': 200,
    'batch_size': 512,
    'loss_function':F.binary_cross_entropy,
    'optimizer_discriminator': torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)),
    'optimizer_generator': torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5,0.999)),
    'save_path': 'training_64',
    'sample_size': 10,
    'plot_every': 10,
    'save_every': 100,
    'features_discriminator' : FEATURES_DISC_DIM,
    'features_generator' : FEATURES_GEN_DIM,
    'noise_dims' : FEATURES_NOISE_DIM
    }
    
    train_dataset = DataLoader(torchvision.datasets.CelebA(root = './data', train = True, 
                                                    download = True, transform = torchvision.transforms.ToTensor()), 
                                                    batch_size = training_params['batch_size'], shuffle=True, 
                                                    num_workers=2, pin_memory=True)

    validation_dataset = DataLoader(torchvision.datasets.CelebA(root = './data', train = False, 
                                                        download = True, transform = torchvision.transforms.ToTensor()),
                                                        batch_size = training_params['batch_size'], shuffle=False, 
                                                        num_workers=2, pin_memory=True)
    
    FIXED_NOISE = torch.randn(training_params['sample_size'], training_params['noise_channels'], 1, 1).to(device)
    
    # Create directory if it doesn't exist
    os.makedirs(training_params['save_path'], exist_ok=True)
    os.makedirs(os.path.join(training_params['save_path'], 'training_images'), exist_ok=True)
    os.makedirs(os.path.join(training_params['save_path'], 'generated_images'), exist_ok=True)
    
    metrics = {
        'l1': lambda output, target: (torch.abs(output - target).sum())
    }
    
    train_results, evaluation_results = train_evaluate(discriminator, generator, device, train_dataset, validation_dataset, training_params, metrics)
    plot_training_results(train_results=train_results, validation_results=evaluation_results, training_params=training_params, metrics=metrics)
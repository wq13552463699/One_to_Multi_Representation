#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:27:41 2021

@author: qiang
"""
import cv2
from CoppeliaSim_UR5_gym_v3 import CoppeliaSim_UR5_gym_v3
from env import GymEnv, EnvBatcher
env = CoppeliaSim_UR5_gym_v3()
from torchvision.utils import make_grid
# from __future__ import print_function
import time
import random
import os
from torch.nn import functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from utils import lineplot
import network
from torchsummary import summary

def preprocess_observation_(observation, bit_depth):
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)

# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)

def _images_to_observation(images, bit_depth):
  # print(images)
  images = torch.tensor(cv2.resize(images, (128,128), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  # images = torch.tensor(cv2.resize(images, (64,64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
  return images.unsqueeze(dim=0)  # Add batch dimension

np.random.seed(42)
random.seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(999)

metrics = {'episodes': [],'observation_loss': []}

gpu_id = 0
device = torch.device("cuda", gpu_id)

netG = network.G().to(device)
netG.apply(network.weights_init)
summary(netG,input_size=(3,128,128))
netD = network.D().to(device)
netD.apply(network.weights_init)
summary(netD,input_size=(3,128,128))

L1_factor = 0
L2_factor = 1
GAN_factor = 0.1

criterion = nn.BCELoss() 


optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999), eps = 1e-8)

# Create a directory for the output files
try:
    os.mkdir('output')
except OSError:
    pass

results_dir = './output'

start_time = time.time()

t = 490000
for epoch in range(100000):
    
    action = torch.from_numpy(env.action_space.sample())  
    next_observation, _ , _ ,_ = env.step(action)

    pic1, pic2 = next_observation
    
    obs1 = _images_to_observation(pic1 ,5)
    obs2 = _images_to_observation(pic2 ,5)
    netD.zero_grad()
    real = Variable(obs2).type('torch.FloatTensor').to(device)
    target = Variable(torch.ones(real.size()[0])).to(device)
    output = netD(real)
    errD_real = criterion(output, target)
    
    profile = Variable(obs1).type('torch.FloatTensor').to(device)
    generated = netG(profile)
    target = Variable(torch.zeros(real.size()[0])).to(device)
    output = netD(generated.detach()) 

    errD_fake = criterion(output, target)
    
    errD = errD_real + errD_fake
    errD.backward()

    optimizerD.step()
    

    netG.zero_grad()
    target = Variable(torch.ones(real.size()[0])).to(device)
    output = netD(generated)
    
    errG_GAN = criterion(output, target)

    obs_err = F.mse_loss(real, generated, reduction='none').sum().mean()#data.cpu().numpy()
    
    errG = (GAN_factor * errG_GAN) + obs_err * 0.1
    
    errG.backward()
    # Update G
    optimizerG.step()

    if epoch % 10 ==0:
        print(GAN_factor * errG_GAN)
        print(errG)

    if epoch % 100 ==0 and epoch != 0:
        obs_err = F.mse_loss(real, generated, reduction='none').sum().mean().data.cpu().numpy()
        metrics['observation_loss'].append(obs_err)
        metrics['episodes'].append(epoch+t)
        lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
        vutils.save_image(profile.data, 'output/%03d_input.jpg' % (epoch+t), normalize=True)
        vutils.save_image(real.data, 'output/%03d_real.jpg' % (epoch+t), normalize=True)
        vutils.save_image(generated.data, 'output/%03d_generated.jpg' % (epoch+t), normalize=True)
    if epoch % 5000 == 0:
        torch.save(netG,'output/netG_%d.pt' % (epoch+t))
        torch.save(netD,'output/netD_%d.pt' % (epoch+t))
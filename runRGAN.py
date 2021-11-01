#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:27:41 2021

@author: qiang
"""

import cv2
def preprocess_observation_(observation, bit_depth):
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)

# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)

def _images_to_observation(images, bit_depth):
  # print(images)
  images = torch.tensor(cv2.resize(images, (128,128), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
  # return images.unsqueeze(dim=0)  # Add batch dimension
  return images

def pre_pos(images):
    image = cv2.resize(images, (128,128), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    image = image.astype(np.float32)
    image /= 255.0
    return image

from CoppeliaSim_RGAN import CoppeliaSim_RGAN
env = CoppeliaSim_RGAN()
env.reset()

# from __future__ import print_function
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

import networkRGAN
from torchsummary import summary

np.random.seed(42)
random.seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(999)

metrics = {'steps': [],'observation_loss': []}


gpu_id = 0
device = torch.device("cuda", gpu_id)


Encoder = torch.load('Encoder_10000.pt')
Decoder = torch.load('Decoder_10000.pt')
Dis = torch.load('Dis_10000.pt')

# Here is where you set how important each component of the loss function is:
L1_factor = 0
L2_factor = 1
GAN_factor = 1
Rec_factor = 0.2

criterion = nn.BCELoss() # Binary cross entropy loss

# Optimizers for the generator and the discriminator (Adam is a fancier version of gradient descent with a few more bells and whistles that is used very often):
optimizerDis = optim.Adam(Dis.parameters(), lr = 0.0001, betas = (0.5, 0.999))
optimizerEncoder = optim.Adam(Encoder.parameters(), lr = 0.0001, betas = (0.5, 0.999), eps = 1e-8)
optimizerDecoder = optim.Adam(Decoder.parameters(), lr = 0.0001, betas = (0.5, 0.999), eps = 1e-8)

def create_batch_data(environment,
                      batch_size,
                      picture_size=128):

    Rel = np.ones((batch_size, 3,picture_size,picture_size))
    Pre = np.ones((batch_size, 3,picture_size,picture_size))
    
    action = torch.from_numpy(env.action_space.sample())  
    next_obs, done= env.step(action)
    
    for i in range(batch_size):
        if done == False:
            obs = next_obs
        elif done == True:
            obs = env.reset()
            # print(1)
            
        pic1, pic2 = obs
        Pre[i] = pre_pos(pic1)
        Rel[i] = pre_pos(pic2)
        
        action = torch.from_numpy(env.action_space.sample())  
        next_obs, done= env.step(action)
        
        
         
    return Rel, Pre

# Create a directory for the output files
direct_name = 'GAN-v3'
batch_size = 1
image_period = 500
model_period = 5000


try:
    os.mkdir(direct_name)
except OSError:
    pass
results_dir = direct_name
t = 0
t_image = int(image_period / batch_size)
t_model = int(model_period / batch_size)
done = False
for _ in range(100000):
    

    Rel, Pre = create_batch_data(env,batch_size)


    # TRAINING THE DISCRIMINATOR
    Dis.zero_grad()
    Rel = torch.tensor(Rel)
    real = Variable(Rel).type('torch.FloatTensor').to(device)
    target = Variable(torch.ones(real.size()[0])).to(device)
    output = Dis(real)
    errD_real = criterion(output, target)
    
    Pre = torch.tensor(Pre)
    predict = Variable(Pre).type('torch.FloatTensor').to(device)
    generated = Encoder(predict)
    generated = Decoder(generated.detach())
    
    target = Variable(torch.zeros(real.size()[0])).to(device)
    output = Dis(generated.detach()) # detach() because we are not training G here
    errD_fake = criterion(output, target)

    errD = errD_real + errD_fake
    errD.backward()    
    optimizerDis.step()

    # TRAINING THE GENERATOR
    Encoder.zero_grad()
    Decoder.zero_grad()
    
    target = Variable(torch.ones(real.size()[0])).to(device)
    output = Dis(generated)

    # G wants to :
    # (a) have the synthetic images be accepted by D (= look like frontal images of people)
    errG_GAN = criterion(output, target)    
    errG_Rec = F.mse_loss(real, generated, reduction='none').sum().mean()#data.cpu().numpy()

    errG = (GAN_factor * errG_GAN) + (errG_Rec * Rec_factor)

    errG.backward()
    # Update G
    optimizerEncoder.step()
    optimizerDecoder.step()

    
    t += 1  
    
    print('t = ' + str(t) + ',  ' + str(errG_Rec/batch_size))
    obs_err = F.mse_loss(real, generated, reduction='none').sum().mean().data.cpu().numpy()
    metrics['observation_loss'].append(errG_Rec.cpu().detach().numpy()/batch_size)
    metrics['steps'].append(t)
    lineplot(metrics['steps'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
    
    if t % t_image == 0:
        vutils.save_image(predict[0].data, direct_name+'/%03d_input.jpg' % (t), normalize=True)
        vutils.save_image(real[0].data, direct_name+'/%03d_real.jpg' % (t), normalize=True)
        vutils.save_image(generated[0].data, direct_name+'/%03d_generated.jpg' % (t), normalize=True)
        
        # Save the pre-trained Generator as well
    if t % t_model == 0:
        tempresults_dir = './'+direct_name+'/%s%d' % ('t',t)
        try:
            os.mkdir(tempresults_dir)
        except OSError:
            pass
        torch.save(Encoder,tempresults_dir+'/Encoder_%d.pt' % (t))
        torch.save(Decoder,tempresults_dir+'/Decoder_%d.pt' % (t))
        torch.save(Dis,tempresults_dir+'/Dis_%d.pt' % (t))       

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:26:44 2021

@author: qiang
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F


def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


''' Generator network for 128x128 RGB images '''
class G(nn.Module):
    
    def __init__(self):
        # super().__init__()
        super(G, self).__init__()
        
        # self.main = nn.Sequential(
        #     # Input HxW = 128x128
        #     nn.Conv2d(3, 16, 4, 2, 1), # Output HxW = 64x64
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(True),
        #     nn.Conv2d(16, 32, 4, 2, 1), # Output HxW = 32x32
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, 4, 2, 1), # Output HxW = 16x16
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 128, 4, 2, 1), # Output HxW = 8x8
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.Conv2d(128, 256, 4, 2, 1), # Output HxW = 4x4
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        #     nn.Conv2d(256, 512, 4, 2, 1), # Output HxW = 2x2
        #     nn.MaxPool2d((2,2)),
        #     # At this point, we arrive at our low D representation vector, which is 512 dimensional.
            
        #     view(-1,512),
        #     nn.Linear(512,1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024,512),
        #     nn.ReLU(True),
            
        #     nn.ConvTranspose2d(512, 256, 4, 1, 0, bias = False), # Output HxW = 4x4
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), # Output HxW = 8x8
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), # Output HxW = 16x16
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False), # Output HxW = 32x32
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, 16, 4, 2, 1, bias = False), # Output HxW = 64x64
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 3, 4, 2, 1, bias = False), # Output HxW = 128x128
        #     nn.Tanh()
        # )
        # self.act_fc = getattr(F, 'relu')
        # self.conv1 = nn.Conv2d(3, 16, 4, 2, 1)
        # self.bat1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, 4, 2, 1)
        # self.bat2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        # self.bat3 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(64, 128, 4, 2, 1)
        # self.bat4 = nn.BatchNorm2d(128)
        # self.conv5 = nn.Conv2d(128, 256, 4, 2, 1)
        # self.bat5 = nn.BatchNorm2d(256)
        # self.conv6 = nn.Conv2d(256, 512, 4, 2, 1)
        # self.Maxpool = nn.MaxPool2d((2,2))
        
        # self.fc1 = nn.Linear(512, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        
        # self.convT1 = nn.ConvTranspose2d(512, 256, 4, 1, 0, bias = False)
        # self.batT1 = nn.BatchNorm2d(256)
        # self.convT2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False)
        # self.batT2 = nn.BatchNorm2d(128)
        # self.convT3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False)
        # self.batT3 = nn.BatchNorm2d(64)
        # self.convT4 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False)
        # self.batT4 = nn.BatchNorm2d(32)
        # self.convT5 = nn.ConvTranspose2d(32, 16, 4, 2, 1, bias = False)
        # self.batT5 = nn.BatchNorm2d(16)
        # self.convT6 = nn.ConvTranspose2d(16, 3, 4, 2, 1, bias = False)
        # self.tanh = nn.Tanh()
        
        self.act_fc = getattr(F, 'relu')
        
        self.conv1 = nn.Conv2d(3, 16, 3, stride = 1, padding = 'same') #128 *128
        self.bat1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 16, 4, stride = 2, padding = 1)#64 *64
        self.bat2 = nn.BatchNorm2d(16)
        
        # self.Maxpool1 = nn.MaxPool2d(kernel_size = (2,2),stride=2) 
        
        self.conv3 = nn.Conv2d(16, 32, 3, stride = 1, padding = 'same')
        self.bat3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 32, 4, stride = 2, padding = 1)#32 * 32
        self.bat4 = nn.BatchNorm2d(32)
        
        # self.Maxpool2 = nn.MaxPool2d(kernel_size = (2,2),stride=2) 
        
        self.conv5 = nn.Conv2d(32, 64, 3, stride = 1, padding = 'same')
        self.bat5 = nn.BatchNorm2d(64)
        
        self.conv6 = nn.Conv2d(64, 64, 3, stride = 1, padding = 'same')
        self.bat6 = nn.BatchNorm2d(64)
        
        self.conv7 = nn.Conv2d(64, 64, 4, stride = 2, padding = 1)#16 * 16
        self.bat7 = nn.BatchNorm2d(64)
        
        # self.Maxpool3 = nn.MaxPool2d(kernel_size = (2,2),stride=2) 
        
        self.conv8 = nn.Conv2d(64, 128, 3, stride = 1, padding = 'same')
        self.bat8 = nn.BatchNorm2d(128)
        
        self.conv9 = nn.Conv2d(128, 128, 3, stride = 1, padding = 'same')
        self.bat9 = nn.BatchNorm2d(128)
        
        self.conv10 = nn.Conv2d(128, 128, 4, stride = 2, padding = 1)#8 * 8 
        self.bat10 = nn.BatchNorm2d(128)
        
        # self.Maxpool4 = nn.MaxPool2d(kernel_size = (2,2),stride=2) 
        
        self.conv11 = nn.Conv2d(128, 256, 3, stride = 1, padding = 'same')
        self.bat11 = nn.BatchNorm2d(256)
        
        self.conv12 = nn.Conv2d(256, 256, 3, stride = 1, padding = 'same')
        self.bat12 = nn.BatchNorm2d(256)
        
        self.conv13 = nn.Conv2d(256, 256, 4, stride = 2, padding = 1)# 4 * 4
        self.bat13 = nn.BatchNorm2d(256)
        
        # self.Maxpool5 = nn.MaxPool2d(kernel_size = (2,2),stride=2) 
        
        self.conv14 = nn.Conv2d(256, 512, 3, stride = 1, padding = 'same')
        self.bat14 = nn.BatchNorm2d(512)
        
        self.conv15 = nn.Conv2d(512, 512, 3, stride = 1, padding = 'same')
        self.bat15 = nn.BatchNorm2d(512)
        
        self.conv16 = nn.Conv2d(512, 512, 4, stride = 2, padding = 1)
        
        self.Maxpool = nn.MaxPool2d(kernel_size = (2,2)) # 1 * 1
        
        
        self.convT1 = nn.ConvTranspose2d(512, 512, 4, stride = 1, padding = 0, bias = False) # Output HxW = 4x4
        self.batT1 = nn.BatchNorm2d(512)
        
        self.convT2 = nn.ConvTranspose2d(512, 512, 3, stride = 1, padding = 1, bias = False)
        self.batT2 = nn.BatchNorm2d(512)
        
        self.convT3 = nn.ConvTranspose2d(512, 256, 4, stride = 2, padding = 1, bias = False) # Output HxW = 88
        self.batT3 = nn.BatchNorm2d(256)
        
        
        self.convT4 = nn.ConvTranspose2d(256, 256, 3, stride = 1, padding = 1, bias = False) 
        self.batT4 = nn.BatchNorm2d(256)
        
        self.convT5 = nn.ConvTranspose2d(256, 256, 3, stride = 1, padding = 1, bias = False) 
        self.batT5 = nn.BatchNorm2d(256)
        
        self.convT6 = nn.ConvTranspose2d(256, 128, 4, stride = 2, padding = 1, bias = False)  # Output HxW = 16 16
        self.batT6 = nn.BatchNorm2d(128)
        
        
        self.convT7 = nn.ConvTranspose2d(128, 128, 3, stride = 1, padding = 1, bias = False) 
        self.batT7 = nn.BatchNorm2d(128)
        
        self.convT8 = nn.ConvTranspose2d(128, 128, 3, stride = 1, padding = 1, bias = False) 
        self.batT8 = nn.BatchNorm2d(128)
        
        self.convT9 = nn.ConvTranspose2d(128, 64, 4, stride = 2, padding = 1, bias = False)  # Output HxW = 32 32
        self.batT9 = nn.BatchNorm2d(64)
        
        
        self.convT10 = nn.ConvTranspose2d(64, 64, 3, stride = 1, padding = 1, bias = False) 
        self.batT10 = nn.BatchNorm2d(64)
        
        self.convT11 = nn.ConvTranspose2d(64, 64, 3, stride = 1, padding = 1, bias = False) 
        self.batT11 = nn.BatchNorm2d(64)
        
        self.convT12 = nn.ConvTranspose2d(64, 32, 4, stride = 2, padding = 1, bias = False)  # Output HxW = 64 64
        self.batT12 = nn.BatchNorm2d(32)
        
        
        self.convT13 = nn.ConvTranspose2d(32, 32, 3, stride = 1, padding = 1, bias = False) 
        self.batT13 = nn.BatchNorm2d(32)
             
        self.convT14 = nn.ConvTranspose2d(32, 16, 4, stride = 2, padding = 1, bias = False)  # Output HxW = 128 128
        self.batT14 = nn.BatchNorm2d(16)
        
        
        self.convT15 = nn.ConvTranspose2d(16, 3, 3, stride = 1, padding = 1, bias = False) 
        
        self.tanh = nn.Tanh()
        
        
    def forward(self, input):
        ##########################################################
        # output = self.main(input)
        # return output
        # hidden = self.conv1(input)
        # hidden = self.bat1(hidden)
        # hidden = self.act_fc(hidden)
        
        # hidden = self.conv2(hidden)
        # hidden = self.bat2(hidden)
        # hidden = self.act_fc(hidden)
        
        # hidden = self.conv3(hidden)
        # hidden = self.bat3(hidden)
        # hidden = self.act_fc(hidden)
        
        # hidden = self.conv4(hidden)
        # hidden = self.bat4(hidden)
        # hidden = self.act_fc(hidden)
        
        # hidden = self.conv5(hidden)
        # hidden = self.bat5(hidden)
        # hidden = self.act_fc(hidden)
        
        # hidden = self.conv6(hidden)
        # hidden = self.Maxpool(hidden)
        
        # # hidden = hidden.view(-1,512)
        # # hidden = self.fc1(hidden)
        # # hidden = self.fc2(hidden)
        # # hidden = hidden.view(-1,512,1,1)
        
        # hidden = self.convT1(hidden)
        # hidden = self.batT1(hidden)
        # hidden = self.act_fc(hidden)

        # hidden = self.convT2(hidden)
        # hidden = self.batT2(hidden)
        # hidden = self.act_fc(hidden)
        
        # hidden = self.convT3(hidden)
        # hidden = self.batT3(hidden)
        # hidden = self.act_fc(hidden)
        
        # hidden = self.convT4(hidden)
        # hidden = self.batT4(hidden)
        # hidden = self.act_fc(hidden)
        
        # hidden = self.convT5(hidden)
        # hidden = self.batT5(hidden)
        # hidden = self.act_fc(hidden)
        
        # hidden = self.convT6(hidden)
        # hidden = self.tanh(hidden)
        ###################################################
        
        hidden = self.conv1(input)
        hidden = self.bat1(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv2(hidden)
        hidden = self.bat2(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv3(hidden)
        hidden = self.bat3(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv4(hidden)
        hidden = self.bat4(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv5(hidden)
        hidden = self.bat5(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv6(hidden)
        hidden = self.bat6(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv7(hidden)
        hidden = self.bat7(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv8(hidden)
        hidden = self.bat8(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv9(hidden)
        hidden = self.bat9(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv10(hidden)
        hidden = self.bat10(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv11(hidden)
        hidden = self.bat11(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv12(hidden)
        hidden = self.bat12(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv13(hidden)
        hidden = self.bat13(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv14(hidden)
        hidden = self.bat14(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv15(hidden)
        hidden = self.bat15(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.conv16(hidden)
        
        hidden = self.Maxpool(hidden)
        
        
        hidden = self.convT1(hidden)
        hidden = self.batT1(hidden)
        hidden = self.act_fc(hidden)

        hidden = self.convT2(hidden)
        hidden = self.batT2(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.convT3(hidden)
        hidden = self.batT3(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.convT4(hidden)
        hidden = self.batT4(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.convT5(hidden)
        hidden = self.batT5(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.convT6(hidden)
        hidden = self.batT6(hidden)
        hidden = self.act_fc(hidden)

        hidden = self.convT7(hidden)
        hidden = self.batT7(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.convT8(hidden)
        hidden = self.batT8(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.convT9(hidden)
        hidden = self.batT9(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.convT10(hidden)
        hidden = self.batT10(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.convT11(hidden)
        hidden = self.batT11(hidden)
        hidden = self.act_fc(hidden)

        hidden = self.convT12(hidden)
        hidden = self.batT12(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.convT13(hidden)
        hidden = self.batT13(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.convT14(hidden)
        hidden = self.batT14(hidden)
        hidden = self.act_fc(hidden)
        
        hidden = self.convT15(hidden)
        
        hidden = self.tanh(hidden)
        
        return hidden

''' Discriminator network for 128x128 RGB images '''
class D(nn.Module):
    
    def __init__(self):
        super(D, self).__init__()
        # (W - F + 2P) / S + 1
        self.main = nn.Sequential(
            
                                  nn.Conv2d(3, 16, 3, stride = 1, padding = 'same'),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(True),
                                  
                                  nn.Conv2d(16, 16, 4, 2, 1),
                                  nn.BatchNorm2d(16),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  
                                  # 64* 64
                                  nn.Conv2d(16, 32, 3, stride = 1, padding = 'same'),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(True),
                                  
                                  nn.Conv2d(32, 32, 4, 2, 1),
                                  nn.BatchNorm2d(32),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  
                                  # 32*32
                                  nn.Conv2d(32, 64, 3, stride = 1, padding = 'same'),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(True),
                                  
                                  nn.Conv2d(64, 64, 3, stride = 1, padding = 'same'),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(True),
                                  
                                  nn.Conv2d(64, 64, 4, 2, 1),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  
                                  # 16*16
                                  nn.Conv2d(64, 128, 3, stride = 1, padding = 'same'),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(True),
                                  
                                  nn.Conv2d(128, 128, 3, stride = 1, padding = 'same'),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(True),
                                  
                                  nn.Conv2d(128, 128, 4, 2, 1),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  
                                  # 8*8
                                  nn.Conv2d(128, 256, 3, stride = 1, padding = 'same'),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  
                                  nn.Conv2d(256, 256, 3, stride = 1, padding = 'same'),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  
                                  nn.Conv2d(256, 256, 4, 2, 1),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  
                                  #4*4
                                  nn.Conv2d(256, 512, 3, stride = 1, padding = 'same'),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(True),
                                  
                                  nn.Conv2d(512, 512, 3, stride = 1, padding = 'same'),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU(True),
                                  
                                  nn.Conv2d(512, 512, 4, 2, 1),
                                  nn.BatchNorm2d(512),
                                  nn.LeakyReLU(0.2, inplace = True),
                                                                    
                                  # 2*2
                                  nn.Conv2d(512, 1, 4, 2, 1, bias = False),
                                  nn.Sigmoid()
                                  )
    
    
    def forward(self, input):
        output = self.main(input)
        return output.view(-1)
# What is output is a float
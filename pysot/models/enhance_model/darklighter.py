import torch
import torch.nn as nn
import math
import numpy as np
from torchvision import transforms
import os
import sys
import argparse
import time

import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import pdb
import cv2

from matplotlib import pyplot as plt

class DarkLighter(nn.Module):

    def __init__(self):
        super(DarkLighter, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 8, 3, 1, 1, bias=True)
        self.e_conv8 = nn.Conv2d(number_f * 2, 8, 3, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = (np.asarray(x.cpu()) / 255.0)
        x = torch.from_numpy(x).float()
        # x = x.permute(2, 0, 1)
        x = x.cuda()

        x1 = self.relu(self.e_conv1(x))
        # print(x1.shape)
        x2 = self.relu(self.e_conv2(x1))
        # print(x2.shape)
        x3 = self.relu(self.e_conv3(torch.cat([x1, x2], 1)))
        x4 = self.relu(self.e_conv4(torch.cat([x2, x3], 1)))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x4, x5], 1)))
        x_n = torch.tanh(self.e_conv8(torch.cat([x4, x5], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 1, dim=1)
        n1, n2, n3, n4, n5, n6, n7, n8 = torch.split(x_n, 1, dim=1)

        x = (x - n1) * (r1 + 1)
        x = (x - n2) * (r2 + 1)
        x = (x - n3) * (r3 + 1)
        enhance_image_1 = (x - n4) * (r4 + 1)
        x = (enhance_image_1 - n5) * (r5 + 1)
        x = (x - n6) * (r6 + 1)
        x = (x - n7) * (r7 + 1)
        enhance_image = (x - n8) * (r8 + 1)

        r = torch.cat([(r1 + 1), (r2 + 1), (r3 + 1), (r4 + 1), (r5 + 1), (r6 + 1), (r7 + 1), (r8 + 1)], 1)
        n = torch.cat([n1, n2, n3, n4, n5, n6, n7, n8], 1)
        # enhance_image2 = enhance_image.view(enhance_image.shape[1],enhance_image.shape[2],enhance_image.shape[3])
        # enhance_image2 = transforms.ToPILImage()(enhance_image2)
        # plt.imshow(enhance_image2)
        return enhance_image*255.0, r, n

    def lowlight(self,image_path):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        data_lowlight = Image.open(image_path)

        data_lowlight = (np.asarray(data_lowlight) / 255.0)

        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight = data_lowlight.permute(2, 0, 1)
        data_lowlight = data_lowlight.cuda().unsqueeze(0)

        start = time.time()
        enhanced_image, _, _ = self.forward(data_lowlight)
        end_time = (time.time() - start)
        print(end_time)
        image_path = image_path.replace('test', 'result')
        result_path = image_path
        if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
            os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))

        torchvision.utils.save_image(enhanced_image, result_path)
        return enhanced_image

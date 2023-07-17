from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.xcorr import xcorr_fast, xcorr_depthwise
from pysot.models.init_weight import init_weights


class Integration(nn.Module):
    def __init__(self):

        super(Integration, self).__init__()
        a = 512
        self.features1 = nn.Conv2d(256*3, a, kernel_size=3, padding=1)
        self.features2 = nn.Conv2d(a, a, kernel_size=3, padding=1)
        self.features3 = nn.Conv2d(a, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = x
        x = self.relu(self.features1(x))
        x = self.relu(self.features2(x))
        x = self.relu(self.features3(x))
        # print(x.shape)
        return x
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adamax
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

from tractseg.libs.PytorchUtils import PytorchUtils
from tractseg.libs.ExpUtils import ExpUtils
from tractseg.models.BaseModel import BaseModel
from tractseg.libs.PytorchUtils import conv2d
from tractseg.libs.PytorchUtils import deconv2d


class UNet_Pytorch_DeepSup_Segmenter(torch.nn.Module):
    def __init__(self, n_input_channels=3, n_classes=7, n_filt=64, batchnorm=False, dropout=False):
        super(UNet_Pytorch_DeepSup_Segmenter, self).__init__()

        self.dropout = dropout

        self.in_channel = n_input_channels
        self.n_classes = n_classes

        self.conv_5 = nn.Conv2d(n_filt, n_classes, kernel_size=1, stride=1, padding=0, bias=True)  # no activation function, because is in LossFunction (...WithLogits)

    def forward(self, output_3_up, expand_4_2):

        conv_5 = self.conv_5(expand_4_2)
        final = output_3_up + conv_5

        return final, F.sigmoid(final)
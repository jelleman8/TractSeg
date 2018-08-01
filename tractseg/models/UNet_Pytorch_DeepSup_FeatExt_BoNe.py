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


class UNet_Pytorch_DeepSup_FeatExt_BoNe(torch.nn.Module):
    def __init__(self, n_input_channels=3, n_classes=7, n_filt=64, batchnorm=False, dropout=False):
        super(UNet_Pytorch_DeepSup_FeatExt_BoNe, self).__init__()

        self.dropout = dropout

        self.in_channel = n_input_channels
        self.n_classes = n_classes

        self.contr_1_1 = conv2d(n_input_channels, n_filt)
        self.contr_1_2 = conv2d(n_filt, n_filt)
        self.pool_1 = nn.MaxPool2d((2, 2))

        self.contr_2_1 = conv2d(n_filt, n_filt * 2)
        self.contr_2_2 = conv2d(n_filt * 2, n_filt * 2)
        self.pool_2 = nn.MaxPool2d((2, 2))

        self.contr_3_1 = conv2d(n_filt * 2, n_filt * 4)
        self.contr_3_2 = conv2d(n_filt * 4, n_filt * 4)
        self.pool_3 = nn.MaxPool2d((2, 2))

        self.contr_4_1 = conv2d(n_filt * 4, n_filt * 8)
        self.contr_4_2 = conv2d(n_filt * 8, n_filt * 8)
        self.pool_4 = nn.MaxPool2d((2, 2))

        self.dropout = nn.Dropout(p=0.4)

        self.encode_1 = conv2d(n_filt * 8, n_filt * 16)
        self.encode_2 = conv2d(n_filt * 16, n_filt * 16)

    def forward(self, inpt):
        contr_1_1 = self.contr_1_1(inpt)
        contr_1_2 = self.contr_1_2(contr_1_1)
        pool_1 = self.pool_1(contr_1_2)

        contr_2_1 = self.contr_2_1(pool_1)
        contr_2_2 = self.contr_2_2(contr_2_1)
        pool_2 = self.pool_2(contr_2_2)

        contr_3_1 = self.contr_3_1(pool_2)
        contr_3_2 = self.contr_3_2(contr_3_1)
        pool_3 = self.pool_3(contr_3_2)

        contr_4_1 = self.contr_4_1(pool_3)
        contr_4_2 = self.contr_4_2(contr_4_1)
        pool_4 = self.pool_4(contr_4_2)

        # pool_4 = self.dropout(pool_4)

        encode_1 = self.encode_1(pool_4)
        encode_2 = self.encode_2(encode_1)

        return encode_2, contr_4_2, contr_3_2, contr_2_2, contr_1_2
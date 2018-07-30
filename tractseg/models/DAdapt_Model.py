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

import torch
import torch.nn as nn
from tractseg.models.UNet_Pytorch_DeepSup_FeatExt import UNet_Pytorch_DeepSup_FeatExt
from tractseg.models.UNet_Pytorch_DeepSup_Segmenter import UNet_Pytorch_DeepSup_Segmenter
from tractseg.models.DomainDiscriminator import DomainDiscriminator
from tractseg.libs.PytorchUtils import ReverseLayerF

class DAdapt_Model(torch.nn.Module):
    def __init__(self, n_input_channels=3, n_classes=7, n_filt=64, batchnorm=False, dropout=False):
        super(DAdapt_Model, self).__init__()

        self.featureExtractor = UNet_Pytorch_DeepSup_FeatExt(n_input_channels=n_input_channels, n_classes=n_classes,
                                                       n_filt=n_filt, batchnorm=batchnorm, dropout=dropout)
        self.segmenter = UNet_Pytorch_DeepSup_Segmenter(n_input_channels=n_input_channels, n_classes=n_classes,
                                                             n_filt=n_filt, batchnorm=batchnorm, dropout=dropout)
        self.domainDiscriminator = DomainDiscriminator(n_input_channels=n_input_channels, n_classes=n_classes,
                                                       n_filt=n_filt, batchnorm=batchnorm, dropout=dropout)

    def forward(self, input, alpha):
        feat_output_1, feat_output_2 = self.featureExtractor(input)
        seg_output, seg_output_sig = self.segmenter(feat_output_1, feat_output_2)
        feat_output_2_rev = ReverseLayerF.apply(feat_output_2, alpha)
        domain_output = self.domainDiscriminator(feat_output_2_rev)

        return seg_output, seg_output_sig, domain_output
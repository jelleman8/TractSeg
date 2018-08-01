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
from tractseg.models.UNet_Pytorch_DeepSup_FeatExt_BoNe import UNet_Pytorch_DeepSup_FeatExt_BoNe
from tractseg.models.UNet_Pytorch_DeepSup_Segmenter_BoNe import UNet_Pytorch_DeepSup_Segmenter_BoNe
from tractseg.models.DomainDiscriminator_BoNe import DomainDiscriminator_BoNe
from tractseg.libs.PytorchUtils import ReverseLayerF

class DAdapt_Model_BoNe(torch.nn.Module):
    def __init__(self, n_input_channels=3, n_classes=74, n_filt=64, batchnorm=False, dropout=False):
        super(DAdapt_Model_BoNe, self).__init__()

        self.featureExtractor = UNet_Pytorch_DeepSup_FeatExt_BoNe(n_input_channels=n_input_channels, n_classes=n_classes,
                                                       n_filt=n_filt, batchnorm=batchnorm, dropout=dropout)
        self.segmenter = UNet_Pytorch_DeepSup_Segmenter_BoNe(n_input_channels=n_input_channels, n_classes=n_classes,
                                                             n_filt=n_filt, batchnorm=batchnorm, dropout=dropout)
        self.domainDiscriminator = DomainDiscriminator_BoNe(n_input_channels=-1, n_classes=2,
                                                       n_filt=64, batchnorm=False, dropout=False, seg_nr_classes=n_classes)

    def forward(self, input, alpha):
        encode_2, contr_4_2, contr_3_2, contr_2_2, contr_1_2 = self.featureExtractor(input)
        seg_output, seg_output_sig = self.segmenter(encode_2, contr_4_2, contr_3_2, contr_2_2, contr_1_2)
        encode_2_rev = ReverseLayerF.apply(encode_2, alpha)
        domain_output = self.domainDiscriminator(encode_2_rev)
        domain_output = torch.squeeze(domain_output)
        return seg_output, seg_output_sig, domain_output
import torch
import torch.nn as nn
import torch.nn.functional as F

from tractseg.libs.PytorchUtils import PytorchUtils
from tractseg.libs.PytorchUtils import conv2d

#All Layers
class DomainDiscriminator_BoNe(torch.nn.Module):
    def __init__(self, n_input_channels=64, n_classes=2, n_filt=64, batchnorm=False, dropout=False, seg_nr_classes=74):
        super(DomainDiscriminator_BoNe, self).__init__()
        self.dropout = dropout
        # self.in_channel = n_input_channels
        self.n_classes = n_classes

        self.conv_1= conv2d(n_filt * 16, n_filt * 16)
        self.conv_2 = conv2d(n_filt * 16, n_filt * 8)

        self.conv_3 = nn.Conv2d(n_filt * 8, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

    # def forward(self, inpt):
    def forward(self, encode_2):
        # encode_2:  [bs, 74, n_filt*16, 5]
        conv_1 = self.conv_1(encode_2)
        conv_2 = self.conv_2(conv_1)

        avg_pool = nn.AvgPool2d(conv_2.size()[2:])(conv_2)  # output: [bs, n_filt*8, 1, 1]
        final = self.conv_5(avg_pool)

        return nn.LogSoftmax()(final)

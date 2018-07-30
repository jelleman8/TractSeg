import torch
import torch.nn as nn
import torch.nn.functional as F

from tractseg.libs.PytorchUtils import PytorchUtils
from tractseg.libs.PytorchUtils import conv2d

class DomainDiscriminator(torch.nn.Module):
    def __init__(self, n_input_channels=64, n_classes=1, n_filt=64, batchnorm=False, dropout=False):
        super(DomainDiscriminator, self).__init__()

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

        #Option A
        # input: [bs, n_filt*8, 5, 5]  if input was 80x80   -> kernel_size=5 only works if input: 80x80
        # self.conv_5 = nn.Conv2d(n_filt * 8, n_classes, kernel_size=5, stride=1, padding=0, bias=True)

        #Option B
        self.conv_5 = nn.Conv2d(n_filt * 8, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

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

        #Option A
        # final = self.conv_5(pool_4)

        # Option B
        avg_pool = nn.AvgPool2d(pool_4, pool_4.size()[2:])  # output: [bs, n_filt*8, 1, 1]
        final = self.conv_5(avg_pool)

        # return final, F.sigmoid(final)
        return F.sigmoid(final)
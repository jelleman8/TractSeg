import torch
import torch.nn as nn
import torch.nn.functional as F

from tractseg.libs.PytorchUtils import PytorchUtils
from tractseg.libs.PytorchUtils import conv2d

#All Layers
class DomainDiscriminator(torch.nn.Module):
    def __init__(self, n_input_channels=64, n_classes=2, n_filt=64, batchnorm=False, dropout=False, seg_nr_classes=74):
        super(DomainDiscriminator, self).__init__()

        self.dropout = dropout

        self.in_channel = n_input_channels
        self.n_classes = n_classes

        self.contr_1_1 = conv2d(n_input_channels + seg_nr_classes, n_filt)
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

    # def forward(self, inpt):
    def forward(self, output_3_up, expand_4_2):

        # output_3_up:  [bs, 74, x, y]
        # expand_4_2:  [bs, 64, x, y]
        concat1 = torch.cat([output_3_up, expand_4_2], 1)

        contr_1_1 = self.contr_1_1(concat1)
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
        avg_pool = nn.AvgPool2d(pool_4.size()[2:])(pool_4)  # output: [bs, n_filt*8, 1, 1]
        final = self.conv_5(avg_pool)

        # return final, F.sigmoid(final)
        return nn.LogSoftmax()(final)



#Simplified
# class DomainDiscriminator(torch.nn.Module):
#     def __init__(self, n_input_channels=64, n_classes=2, n_filt=64, batchnorm=False, dropout=False):
#         super(DomainDiscriminator, self).__init__()
#
#         self.main = nn.Sequential(
#             # input is (nc) x 80 x 80
#             nn.Conv2d(n_input_channels, n_filt, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 40 x 40
#             nn.Conv2d(n_filt, n_filt * 2, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 20 x 20
#             nn.Conv2d(n_filt * 2, n_filt * 4, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 10 x 10
#             nn.Conv2d(n_filt * 4, n_filt * 8, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 5 x 5
#             nn.Conv2d(n_filt * 8, n_classes, 5, 1, 0, bias=False),
#             nn.LogSoftmax()
#         )
#
#     def forward(self, inpt):
#         return self.main(inpt)
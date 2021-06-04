from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class SharedDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(SharedDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.logsigma = nn.Parameter(torch.FloatTensor([2, 2]))

        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            setattr(self, 'upconv_{}_0'.format(i), ConvBlock(num_ch_in, num_ch_out))

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            setattr(self, 'upconv_{}_1'.format(i), ConvBlock(num_ch_in, num_ch_out))
    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = getattr(self, 'upconv_{}_0'.format(i))(x)
            if i > 2:
               x = [x]
            else:
               x = [upsample(x)]
            self.outputs[('upconv', i, 0)] = x[0]


            if self.use_skips and i > 0:
                x+= [input_features[i-1]]
            x = torch.cat(x, 1)
            x = getattr(self, 'upconv_{}_1'.format(i))(x)
            self.outputs[('upconv', i, 1)] = x
        return self.outputs, self.logsigma

class AttentionDispDecoder(nn.Module):
    def __init__(self, scale, num_output_channels=1):
        super(AttentionDispDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.scale = scale

        setattr(self, 'convs', Conv3x3(self.num_ch_dec[self.scale], self.num_output_channels))

        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        out1 = getattr(self, 'convs')(input)
        out = self.sigmoid(out1)
        return out
class AttentionInsDecoder(nn.Module):
    def __init__(self, scale, num_output_channels=1):
        super(AttentionInsDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.num_ch_dec = np.array([16,32, 64, 128, 256])
        self. scale = scale

        setattr(self, 'convs', Conv3x3(self.num_ch_dec[self.scale], self.num_output_channels))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, input):
        out = getattr(self, 'convs')(input)
        out = self.tanh(out)
        return out

class AttentionDecoder(nn.Module):
    def __init__(self, ch1, ch2, do_upsample=False):
        super(AttentionDecoder, self).__init__()

        self.conv1 = attentionConv3x3(ch1, ch2)
        self.conv2 = Conv1x1(ch1, ch2, use_relu=True)
        self.conv3 = Conv1x1(ch2, ch2, use_relu=False)
        self.do_upsample = do_upsample

    def forward(self, inputs):
        in1 = inputs[0]
        in2 = inputs[1]
        in3 = inputs[2]
        if self.do_upsample:
            in1 = upsample(in1)
        x = self.conv1(in1)
        x = [x, in2]
        x = torch.cat(x,1)
        x = self.conv2(x)
        
        x = self.conv3(x)
        x = x * in3
        return x


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        
        setattr(self, 'squeeze', nn.Conv2d(self.num_ch_enc[-1], 256, 1))
        setattr(self, 'pose_0', nn.Conv2d(num_input_features * 256, 256, 3, stride, 1))
        setattr(self, 'pose_1', nn.Conv2d(256, 256, 3, stride, 1))
        setattr(self, 'pose_2', nn.Conv2d(256, 6 * num_frames_to_predict_for, 1))



        self.relu = nn.ReLU()

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(getattr(self, 'squeeze')(f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = getattr(self, 'pose_{}'.format(i))(out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

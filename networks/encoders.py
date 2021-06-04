from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from layers import *

class ResNetEightLayers(models.ResNet):
    def __init__(self, block, layers, dilate_scale=8):
        super(ResNetEightLayers, self).__init__(block, [2, 2, 2, 2])
        from functools import partial
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer11 = self._make_layer(block, 64, layers[0])
        self.layer12 = self._make_layer(block, 64, layers[1])
        self.layer21 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer22 = self._make_layer(block, 128, layers[3])
        self.layer31 = self._make_layer(block, 256, layers[4], stride=2)
        self.layer32 = self._make_layer(block, 256, layers[5])
        self.layer41 = self._make_layer(block, 512, layers[6], stride=2)
        self.layer42 = self._make_layer(block, 512, layers[7])
        if dilate_scale == 8:
            self.layer31.apply(partial(self._nostride_dilate, dilate=2))
            self.layer32.apply(partial(self._nostride_dilate, dilate=2))
            self.layer41.apply(partial(self._nostride_dilate, dilate=4))
            self.layer42.apply(partial(self._nostride_dilate, dilate=4))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2,2):
                m.stride = (1,1)
                if m.kernel_size == (3,3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3,3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, dilate_scale=8, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        from functools import partial
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if dilate_scale == 8:
            self.layer3.apply(partial(self._nostride_dilate, dilate=2))
            self.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            self.layer4.apply(partial(self._nostride_dilate, dilate=2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2,2):
                m.stride = (1,1)
                if m.kernel_size == (3,3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3,3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
class ResNetDilatedSingleImage(models.ResNet):

    def __init__(self, block, layers, dilate_scale=8, num_classes=1000, num_input_images=1):
        super(ResNetDilatedSingleImage, self).__init__(block, layers)
        from functools import partial
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if dilate_scale == 8:
            self.layer3.apply(partial(self._nostride_dilate, dilate=2))
            self.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            self.layer4.apply(partial(self._nostride_dilate, dilate=2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2,2):
                m.stride = (1,1)
                if m.kernel_size == (3,3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3,3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    model = ResNetMultiImageInput(
        models.resnet.BasicBlock, blocks, dilate_scale=8, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model

def resnet_dilated(num_layers, pretrained=False, dilate_scale=8):
    assert num_layers in [18, 50], 'can only run with 18 or 50 layer resnet'
 
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    model = ResNetDilatedSingleImage(models.resnet.BasicBlock, blocks, dilate_scale=8)

    if pretrained:
       loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
       model.load_state_dict(loaded)
    return model

class ResnetEncoderMtan(nn.Module):

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoderMtan, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        
        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        blocks = [1, 1, 1, 1, 1, 1, 1, 1]
        self.encoder = ResNetEightLayers(models.resnet.BasicBlock, blocks)
        if pretrained:
            loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
            loaded_2 = loaded.copy()
            for key, value in loaded.items():
                if 'layer1.0' in key:
                    new_key = key.replace('layer1.0', 'layer11.0')
                    loaded_2[new_key] = loaded[key]
                elif 'layer1.1' in key:
                    new_key = key.replace('layer1.1', 'layer12.0')
                    loaded_2[new_key] = loaded[key]
                elif 'layer2.0' in key:
                    new_key = key.replace('layer2.0', 'layer21.0')
                    loaded_2[new_key] = loaded[key]
                elif 'layer2.1' in key:
                    new_key = key.replace('layer2.1', 'layer22.0')
                    loaded_2[new_key] = loaded[key]
                elif 'layer3.0' in key:
                    new_key = key.replace('layer3.0', 'layer31.0')
                    loaded_2[new_key] = loaded[key]
                elif 'layer3.1' in key:
                    new_key = key.replace('layer3.1', 'layer32.0')
                    loaded_2[new_key] = loaded[key]
                elif 'layer4.0' in key:
                    new_key = key.replace('layer4.0', 'layer41.0')
                    loaded_2[new_key] = loaded[key]
                elif 'layer4.1' in key:
                    new_key = key.replace('layer4.1', 'layer42.0')
                    loaded_2[new_key] = loaded[key]
            self.encoder.load_state_dict(loaded_2)

    def forward(self, input_image):
        x = (input_image - 0.45) / 0.255
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        o1 = self.encoder.relu(x)
        o2 = self.encoder.layer11(self.encoder.maxpool(o1))
        o3 = self.encoder.layer12(o2)
        o4 = self.encoder.layer21(o3)
        o5 = self.encoder.layer22(o4)
        o6 = self.encoder.layer31(o5)
        o7 = self.encoder.layer32(o6)
        o8 = self.encoder.layer41(o7)
        o9 = self.encoder.layer42(o8)

        return o1, o2, o3, o4, o5, o6, o7, o8, o9

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.layer1bn1 = None
        self.layer2bn1 = None
        self.layer3bn1 = None
        self.layer4bn1 = None

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnet_dilated(num_layers, pretrained, dilate_scale=8)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        o1 = self.encoder.relu(x)
        o2 = self.encoder.layer1(self.encoder.maxpool(o1))
        o3 = self.encoder.layer2(o2)
        o4 = self.encoder.layer3(o3)
        o5 = self.encoder.layer4(o4)
        return o1, o2, o3, o4, o5

class AttentionEncoder(nn.Module):

    def __init__(self, ch1, ch2, do_pool = True):
        super(AttentionEncoder, self, ).__init__()

        self.conv1 = Conv1x1(ch1, ch2, use_relu=True)
        self.conv2 = Conv1x1(ch2, ch2, use_relu=False)
        self.conv3 = attentionConv3x3(ch2, ch2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.do_pool=do_pool
    def forward(self, inputs):
        in1 = inputs[0]
        in2 = inputs[1]

        out = self.conv1(in1)
        out = self.conv2(out)
        mask = out
        out = out * in2
        out = self.conv3(out)
        if self.do_pool:
            out = self.pool(out)
        return out


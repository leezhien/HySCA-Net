import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

# 假设 test_xlstm 模块在同一目录下
from test_xlstm import *
NEW_CHANNELS = 64
NUM_CLASSES = 9


class ELA(nn.Module):
    def __init__(self, in_channels, phi):
        super(ELA, self).__init__()

        Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels // 8, 'L': in_channels // 8}[phi]
        num_groups = {'T': 32, 'B': 16, 'S': 16, 'L': 16}[phi]
        pad = Kernel_size // 2

        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        self.GN = nn.GroupNorm(num_groups, in_channels)
        self.sigmoid = nn.Sigmoid()

        self.xlstm_stack1_cfg1 = xLSTMBlockStack(cfg1).to("cuda:0")
        self.xlstm_stack1_cfg2 = xLSTMBlockStack(cfg2).to("cuda:0")

    def forward(self, input):
        b, c, h, w = input.size()

        x1_ls2 = input.reshape(b, h, c * w)
        y2 = self.xlstm_stack1_cfg2(x1_ls2)
        y2 = y2.reshape(b, c, w, h)

        x1_ls1 = input.reshape(b, w, c * h)
        y1 = self.xlstm_stack1_cfg1(x1_ls1)
        y1 = y1.reshape(b, c, w, h)

        x_h = torch.mean(y2, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(y1, dim=2, keepdim=True).view(b, c, w)

        x_h = self.con1(x_h)
        x_w = self.con1(x_w)

        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)

        return x_h * x_w * input


class SELayer(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1)
        self.batchnorm1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y_out = self.fc1(y)

        y1 = self.conv1(x)
        y1 = self.batchnorm1(y1)
        y1 = self.leaky_relu(y1)

        y1 = self.conv2(y1)
        y1 = self.batchnorm2(y1)
        y1 = self.leaky_relu(y1)

        y1 = self.conv3(y1)
        y1 = self.batchnorm3(y1)

        weight_left = y_out
        weight_right = self.sigmoid(y1)

        return y1 * weight_left * weight_right + x


class FullModel1(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)

    def __init__(self):
        super().__init__()

        self.resblock = SELayer(NEW_CHANNELS)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(NEW_CHANNELS, NUM_CLASSES)

        self.conv1 = nn.Conv2d(103, NEW_CHANNELS, 3)
        self.bn1 = nn.BatchNorm2d(NEW_CHANNELS)
        self.bn2 = nn.BatchNorm2d(NEW_CHANNELS)

        self.dropout1 = nn.Dropout(p=0.8)
        self.ela = ELA(in_channels=NEW_CHANNELS, phi='T')

        self.apply(self.weight_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.resblock(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.ela(x)

        x = self.dropout1(x)
        x = self.avgpool(x)
        feature = self.flatten(x)

        output = self.classifier(feature)
        return feature, output
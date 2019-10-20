# coding = utf-8

import os
import torch
import torch.nn as nn 

BN_MOMENTUM = 0.1


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.conv2d(inplanes, planes, kernel_size, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.conv2d(planes, planes, kernel_size, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out
    

class HRModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, in_channels, out_channels):
        super(HRModule, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_branches = num_branches

        self.branches = self._make_branches(num_branches, blocks, num_blocks, out_channels)
        self.relu = nn.ReLU(False)

    def _make_branches(self, num_branches, blocks, num_blocks, out_channels):
        '''
        pesudo code
        for i in range(num_branches):
            x[i] = self._make_one_branches(branch_index=i)
        return x
        '''
        pass

    def _make_one_branches(self, branch_index, block, num_blocks, num_channels):
        '''
        pesudo code
        for i in range (num_blocks):
            x = block(x, num_channels)
        return x
        '''
        pass

    def _make_fuse_layers(self, num_branches, block, num_blocks, num_channels):
        '''
        pesudo code
        for i in range(num_branches)
            for j in range(num_branches):
                if i != j:
                    do downsample or upsample
                y += x[j]
        '''
        pass

    def forward(self, x):
        pass

class HRNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(HRNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.relu = nn.ReLU(inplace=True)

        # First layer

        # Stages
        self.stage1 = self._make_stage()

        self.stage2 = self._make_stage()

        self.stage3 = self._make_stage()

        self.stage4 = self._make_stage()

        # classfication head
        self.final_layer = self._make_head()
        self.classifier = nn.Linear(2048, 10)


    def _make_head(self, parameter_list):
        '''
        The function from feature maps to classifier.
        pesudo code:

        '''
        pass

    def _make_stage(self, parameter_list):
        '''
        Nothing
        '''
        pass

    def init_weights(self):
        '''
        Initialized weights without pretrained model.
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # The resolution of images in cifar-10 is only 32*32*3.
        # May reduce the first two convolution layer to save the resolution.
        x = self.layer1(x)
        # stage 1
        
        # stage 2

        # stage 3

        # stage 4

        # classification head

        y = self.classifier(y)

        return y

class HRNet_LSTM(nn.Module):
    '''
    IDEA:
    for each branch of results, we simply use num_branches LSTM cell to process different resolution of feature maps and try to make use on position-sensitive tasks.
    '''
    pass

def cls_net(config, **kwargs):
    model = HRNet(config, **kwargs)
    model.init_weights()
    return model
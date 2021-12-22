import torch
import torch.nn as nn
from torch.nn import functional as func
from torch.nn import functional as F
from torch.nn.init import kaiming_normal
import numpy as np

__all__ = [
     'runet'
]



class DepthCNN2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=(1,1), padding=(0,0), stride=(1,1)):
        super(DepthCNN2d, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        
        self.weight_tensor_depth = in_channels * kernel_size[0] * kernel_size[1]
        self.weights = nn.Parameter(torch.empty((1, self.weight_tensor_depth, 1, out_channels),
                                   requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.empty((out_channels),
                                requires_grad=True, dtype=torch.float))

        torch.nn.init.kaiming_uniform_(self.weights)
        torch.nn.init.constant_(self.bias, 0)
        
    def forward(self, input, g_buffer):
        depth = 2 * (g_buffer[:, -1:, :, :] - 0.5)
        input_unf = torch.nn.functional.unfold(input, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

        depth_feature = torch.nn.functional.unfold(depth, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        depth_feature = depth_feature.view(depth_feature.size(0), depth.size(1), -1, depth_feature.size(2))
        depth_feature = (depth_feature - depth.view(depth.size(0), depth.size(1), 1, -1))
        depth_feature = torch.pow(depth_feature, 2)
        depth_feature = depth_feature.sum(dim = 1)

        weight_feature = torch.exp(-1 * depth_feature)
        weight_feature = 9.0 * weight_feature / weight_feature.sum(dim = 1).unsqueeze(1)


        filter_input_unf = input_unf.view(input_unf.size(0), -1, weight_feature.size(1), input_unf.size(2)) * weight_feature.unsqueeze(1)
        input_unf = filter_input_unf.view(input_unf.size(0), input_unf.size(1), input_unf.size(2))
        out_unf = input_unf.transpose(1, 2).matmul(self.weights.view(self.out_channels, -1).t()).transpose(1, 2)
        out = torch.nn.functional.fold(out_unf, (int(depth.shape[2]), int(depth.shape[3])), (1, 1))
        out = out + self.bias.view(1,-1,1,1)

        return out

class RecurrentBlock(nn.Module):

    def __init__(self, input_nc, output_nc, downsampling=False, bottleneck=False, upsampling=False, non_local=False, depth_cnn=False):
        super(RecurrentBlock, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.downsampling = downsampling
        self.upsampling = upsampling
        self.bottleneck = bottleneck
        self.non_local = non_local
        self.depth_cnn = depth_cnn
        self.hidden = None

        if self.downsampling and self.depth_cnn:
            self.relu = nn.LeakyReLU(negative_slope=0.1)
            self.l1 = DepthCNN2d(input_nc, output_nc, kernel_size = (3, 3), padding = (1,1))
        elif self.upsampling and self.depth_cnn:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.relu = nn.LeakyReLU(negative_slope=0.1)
            self.l1 = DepthCNN2d(2 * input_nc, output_nc, kernel_size = (3, 3), padding = (1,1))
        elif self.bottleneck:
            if self.depth_cnn:
                self.relu = nn.LeakyReLU(negative_slope=0.1)
                self.l1 = DepthCNN2d(input_nc, output_nc, kernel_size = (3, 3), padding = (1,1))

    def forward(self, inp, depth):
        if self.downsampling and self.depth_cnn:
            op1 = self.relu(self.l1(inp, depth))
            return op1
        elif self.upsampling and self.depth_cnn:
            up_inp = self.up(inp)
            up_depth = self.up(depth)
            op1 = self.relu(self.l1(up_inp, up_depth))
            return op1
        elif self.bottleneck:
            if self.depth_cnn:
                op1 = self.relu(self.l1(inp, depth))
                return op1


class RecurrentAE(nn.Module):

    def __init__(self):
        super(RecurrentAE, self).__init__()
        self.d1 = RecurrentBlock(input_nc=9, output_nc=16, downsampling=True, depth_cnn=True)
        self.d2 = RecurrentBlock(input_nc=16, output_nc=32, downsampling=True, depth_cnn=True)
        self.d3 = RecurrentBlock(input_nc=32, output_nc=64, downsampling=True, depth_cnn=True)
        
        self.bottleneck = RecurrentBlock(input_nc=64, output_nc=64, bottleneck=True, non_local=False, depth_cnn=True)

        self.u3 = RecurrentBlock(input_nc=64, output_nc=32, upsampling=True, depth_cnn=True)
        self.u2 = RecurrentBlock(input_nc=32, output_nc=16, upsampling=True, depth_cnn=True)
        self.u1 = RecurrentBlock(input_nc=16, output_nc=3, upsampling=True, depth_cnn=True)

        self.inp = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, input):
        inp = input
        depth = input[:, -1:, :, :]
        
        d1 = func.max_pool2d(input=self.d1(inp, depth), kernel_size=2, ceil_mode=True)
        depth_2 = func.max_pool2d(input=depth, kernel_size=2, ceil_mode=True)
        d2 = func.max_pool2d(input=self.d2(d1, depth_2), kernel_size=2, ceil_mode=True)
        depth_3 = func.max_pool2d(input=depth_2, kernel_size=2, ceil_mode=True)
        d3 = func.max_pool2d(input=self.d3(d2, depth_3), kernel_size=2, ceil_mode=True)
        depth_4 = func.max_pool2d(input=depth_3, kernel_size=2, ceil_mode=True)
        
        b = self.bottleneck(d3, depth_4)

        u3 = self.u3(torch.cat((b, d3), dim=1), depth_4)
        u2 = self.u2(torch.cat((u3, d2), dim=1), depth_3)
        u1 = self.u1(torch.cat((u2, d1), dim=1), depth_2)

        return u1

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

def runet(data=None):
    model = RecurrentAE()
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model

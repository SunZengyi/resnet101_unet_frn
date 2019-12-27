import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class FilterResponseNormalization(nn.Module):
    
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        """
        Input Variables:
        ----------------
            num_features: A integer indicating the number of input feature dimensions.
            eps: A scalar constant or learnable variable.
            learnable_eps: A bool value indicating whether the eps is learnable.
        """
        super(FilterResponseNormalization, self).__init__()
        self.eps = nn.Parameter(torch.Tensor([eps]))
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.tau = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.reset_parameters()
    
    def forward(self, x):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """
        nu2 = torch.pow(x, 2).mean(dim=(2, 3), keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)


def conv3x3(in_planes, out_planes, stride=1):
    '''3x3 conv with padding'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes,kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self),__init__()
        if norm_layer is None:
            norm_layer = FilterResponseNormalization
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out) 
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = FilterResponseNormalization
        
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        
        return out 
        
class ResNet(nn.Module):
    def __init__(self, block, layers, num_class=1, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = FilterResponseNormalization
        
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                FilterResponseNormalization(planes * block.expansion), 
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        input = x
        x = self.conv1(x)  # 1/2  128
        x = self.bn1(x)
        x = self.relu(x)
        f1 = self.maxpool(x) # 1/4 64
        
        f2 = self.layer1(f1) # 1/4 64
        f3 = self.layer2(f2) # 1/8 32
        f4 = self.layer3(f3) # 1/16 16
        f5 = self.layer4(f4) # 1/32 8
        
        return [ x, f2, f3, f4, f5, input]

def Resnet101():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model



class UNetConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, normal_layer=None):
        super(UNetConvBlock, self).__init__()
        if normal_layer is None:
            normal_layer = FilterResponseNormalization

        self.conv1 = conv3x3(in_planes, out_planes)
        self.bn1 = normal_layer(out_planes)
        
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = normal_layer(out_planes)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, 
                 up_conv_in_channels=None,up_conv_out_channels=None, up_mode='upconv'):
        super(UNetUpBlock, self).__init__()
        
        if up_conv_in_channels == None:
            up_conv_in_channels = in_chans
        if up_conv_out_channels == None:
            up_conv_out_channels = out_chans
            
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_chans, out_chans)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class ResNetUNet(nn.Module):
    def __init__(
        self,
        n_classes=1,
        norm_layer=None,
        up_mode='upconv',
    ):
        super(ResNetUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        if norm_layer is None:
            norm_layer = FilterResponseNormalization
        
        self.encoder = Resnet101()
        
        self.decoder1 = UNetUpBlock(2048, 1024)
        self.decoder2 = UNetUpBlock(1024, 512)
        self.decoder3 = UNetUpBlock(512, 256)
        self.decoder4 = UNetUpBlock(in_chans=128 + 64,out_chans=128,
                                    up_conv_in_channels=256, up_conv_out_channels=128)
        
        """
        Output Coder
        """
        
        self.decoder5 = UNetUpBlock(in_chans=64 + 3,out_chans=64,
                                    up_conv_in_channels=128, up_conv_out_channels=64)


        self.last = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        
        encoder_output = self.encoder(x)
        encode0 = encoder_output[0]      # [1, 64, 128, 128]
        encode1 = encoder_output[1]      # [1, 256, 64, 64]
        encode2 = encoder_output[2]      # [1, 512, 32, 32]
        encode3 = encoder_output[3]      # [1, 1024, 16, 16]
        encode4 = encoder_output[4]      # [1, 2048, 8, 8]
        input_img = encoder_output[5]    # [1, 3, 256, 256]
        
        decode1 = self.decoder1(encode4, encode3)    # [1, 1024, 16, 16]    
        decode2 = self.decoder2(decode1, encode2)    # [1, 512, 32, 32]
        decode3 = self.decoder3(decode2, encode1)    # [1, 256, 64, 64]
        decode4 = self.decoder4(decode3, encode0)    # [1, 128, 128, 128]
        
        decode5 = self.decoder5(decode4, input_img)   # [1, 64, 256, 256]
    
        out = self.last(decode5)
        
        return out

def main():
    x = torch.randn((1,3, 256,256))
    unet = ResNetUNet()
    unet.eval()
    y_unet = unet(x)
    print(y_unet.size())

if __name__ == '__main__':
    main()

    
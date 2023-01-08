import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from model.MobileNetV2 import mobilenet_v2
from torch.nn import Parameter


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.PReLU(out_channel))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


# Channel-wise Correlation
class CCorrM(nn.Module):
    def __init__(self, all_channel=24, all_dim=128):
        super(CCorrM, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False) #weight
        self.channel = all_channel
        self.dim = all_dim * all_dim
        self.conv1 = DSConv3x3(all_channel, all_channel, stride=1)
        self.conv2 = DSConv3x3(all_channel, all_channel, stride=1)

    def forward(self, exemplar, query):  # exemplar: f1, query: f2
        fea_size = query.size()[2:]
        exemplar = F.interpolate(exemplar, size=fea_size, mode="bilinear", align_corners=True)
        all_dim = fea_size[0] * fea_size[1]
        exemplar_flat = exemplar.view(-1, self.channel, all_dim)  # N,C1,H,W -> N,C1,H*W
        query_flat = query.view(-1, self.channel, all_dim)  # N,C2,H,W -> N,C2,H*W
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batchsize x dim x num, N,H*W,C1
        exemplar_corr = self.linear_e(exemplar_t)  # batchsize x dim x num, N,H*W,C1
        A = torch.bmm(query_flat, exemplar_corr)  # ChannelCorrelation: N,C2,H*W x N,H*W,C1 = N,C2,C1

        A1 = F.softmax(A.clone(), dim=2)  # N,C2,C1. dim=2 is row-wise norm. Sr
        B = F.softmax(torch.transpose(A, 1, 2), dim=2)  # N,C1,C2 column-wise norm. Sc
        query_att = torch.bmm(A1, exemplar_flat).contiguous()  # N,C2,C1 X N,C1,H*W = N,C2,H*W
        exemplar_att = torch.bmm(B, query_flat).contiguous()  # N,C1,C2 X N,C2,H*W = N,C1,H*W

        exemplar_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])  # N,C1,H*W -> N,C1,H,W
        exemplar_out = self.conv1(exemplar_att + exemplar)

        query_att = query_att.view(-1, self.channel, fea_size[0], fea_size[1])  # N,C2,H*W -> N,C2,H,W
        query_out = self.conv1(query_att + query)

        return exemplar_out, query_out


# Edge-based Enhancement Unit (EEU)
class EEU(nn.Module):
    def __init__(self, in_channel):
        super(EEU, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.PReLU = nn.PReLU(in_channel)

    def forward(self, x):
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        out = weight * x + x
        return self.PReLU(edge), out


# Edge Self-Alignment Module (ESAM)
class ESAM(nn.Module):
    def __init__(self, channel1=16, channel2=24):
        super(ESAM, self).__init__()

        self.smooth1 = DSConv3x3(channel1, channel2, stride=1, dilation=1)  # 16channel-> 24channel

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.smooth2 = DSConv3x3(channel2, channel2, stride=1, dilation=1)  # 24channel-> 24channel

        self.eeu1 = EEU(channel2)
        self.eeu2 = EEU(channel2)
        self.ChannelCorrelation = CCorrM(channel2, 128)

    def forward(self, x1, x2):  # x1 16*144*14; x2 24*72*72

        x1_1 = self.smooth1(x1)
        edge1, x1_2 = self.eeu1(x1_1)

        x2_1 = self.smooth2(self.upsample2(x2))
        edge2, x2_2 = self.eeu2(x2_1)

        # Channel-wise Correlation
        x1_out, x2_out = self.ChannelCorrelation(x1_2, x2_2)

        return edge1, edge2, torch.cat([x1_out, x2_out], 1)  # (24*2)*144*144


# Dynamic Semantic Matching Module (DSMM)
class DSMM(nn.Module):
    def __init__(self, channel4=96, channel3=32):
        super(DSMM, self).__init__()

        self.fuse4 = convbnrelu(channel4, channel4, k=1, s=1, p=0, relu=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.smooth4 = DSConv3x3(channel4, channel4, stride=1, dilation=1)  # 96channel-> 96channel

        self.fuse3 = convbnrelu(channel3, channel3, k=1, s=1, p=0, relu=True)
        self.smooth3 = DSConv3x3(channel3, channel4, stride=1, dilation=1)  # 32channel-> 96channel
        self.ChannelCorrelation = CCorrM(channel4, 32)

    def forward(self, x4, k4, x3, k3):  # x4:96*18*18 k4:96*5*5; x3:32*36*36 k3:32*5*5
        B4, C4, H4, W4 = k4.size()
        B3, C3, H3, W3 = k3.size()

        x_B4, x_C4, x_H4, x_W4 = x4.size()  # 8*96*18*18
        x_B3, x_C3, x_H3, x_W3 = x3.size()  # 8*32*36*36

        x4_new = x4.clone()
        x3_new = x3.clone()
        # k4 = k4.view(C4, 1, H4, W4)
        # k3 = k3.view(C3, 1, H3, W3)
        for i in range(1, B4):
            kernel4 = k4[i, :, :, :]
            kernel3 = k3[i, :, :, :]
            kernel4 = kernel4.view(C4, 1, H4, W4)
            kernel3 = kernel3.view(C3, 1, H3, W3)
            # DDconv
            x4_r1 = F.conv2d(x4[i, :, :, :].view(1, C4, x_H4, x_W4), kernel4, stride=1, padding=2, dilation=1,
                             groups=C4)
            x4_r2 = F.conv2d(x4[i, :, :, :].view(1, C4, x_H4, x_W4), kernel4, stride=1, padding=4, dilation=2,
                             groups=C4)
            x4_r3 = F.conv2d(x4[i, :, :, :].view(1, C4, x_H4, x_W4), kernel4, stride=1, padding=6, dilation=3,
                             groups=C4)
            x4_new[i, :, :, :] = x4_r1 + x4_r2 + x4_r3

            # DDconv
            x3_r1 = F.conv2d(x3[i, :, :, :].view(1, C3, x_H3, x_W3), kernel3, stride=1, padding=2, dilation=1,
                             groups=C3)
            x3_r2 = F.conv2d(x3[i, :, :, :].view(1, C3, x_H3, x_W3), kernel3, stride=1, padding=4, dilation=2,
                             groups=C3)
            x3_r3 = F.conv2d(x3[i, :, :, :].view(1, C3, x_H3, x_W3), kernel3, stride=1, padding=6, dilation=3,
                             groups=C3)
            x3_new[i, :, :, :] = x3_r1 + x3_r2 + x3_r3
        # Pconv
        x4_all = self.fuse4(x4_new)
        x4_smooth = self.smooth4(self.upsample2(x4_all))
        # Pconv
        x3_all = self.fuse3(x3_new)
        x3_smooth = self.smooth3(x3_all)

        # Channel-wise Correlation
        x3_out, x4_out = self.ChannelCorrelation(x3_smooth, x4_smooth)

        return torch.cat([x3_out, x4_out], 1)  # (96*2)*32*32


class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.conv(x)


class prediction_decoder(nn.Module):
    def __init__(self, channel5=320, channel34=192, channel12=48):
        super(prediction_decoder, self).__init__()
        # 9*9
        self.decoder5 = nn.Sequential(
            DSConv3x3(channel5, channel5, stride=1),
            DSConv3x3(channel5, channel5, stride=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),  # 36*36
            DSConv3x3(channel5, channel34, stride=1)
        )
        self.s5 = SalHead(channel34)  # 36*36

        # 36*36
        self.decoder34 = nn.Sequential(
            DSConv3x3(channel34 * 2, channel34, stride=1),
            DSConv3x3(channel34, channel34, stride=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),  # 144*144
            DSConv3x3(channel34, channel12, stride=1)
        )
        self.s34 = SalHead(channel12)  # 144*144

        # 144*144
        self.decoder12 = nn.Sequential(
            DSConv3x3(channel12 * 2, channel12, stride=1),
            DSConv3x3(channel12, channel12, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 288*288
            DSConv3x3(channel12, channel12, stride=1)
        )
        self.s12 = SalHead(channel12)

    def forward(self, x5, x34, x12):
        x5_decoder = self.decoder5(x5)
        s5 = self.s5(x5_decoder)

        x34_decoder = self.decoder34(torch.cat([x5_decoder, x34], 1))
        s34 = self.s34(x34_decoder)

        x12_decoder = self.decoder12(torch.cat([x34_decoder, x12], 1))
        s12 = self.s12(x12_decoder)

        return s12, s34, s5


class SeaNet(nn.Module):
    def __init__(self, pretrained=True, channel=128):
        super(SeaNet, self).__init__()
        # Backbone model
        self.backbone = mobilenet_v2(pretrained)
        # input 256*256*3
        # conv1 128*128*16
        # conv2 64*64*24
        # conv3 32*32*32
        # conv4 16*16*96
        # conv5 8*8*320

        # Semantic Knowledge Compression(SKC) unit, k3 and k4
        self.conv5_conv4 = DSConv3x3(320, 96, stride=1)
        self.conv5_conv3 = DSConv3x3(320, 32, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(5)

        self.dsmm = DSMM(96, 32)
        self.esam = ESAM(16, 24)

        self.prediction_decoder = prediction_decoder(320, 192, 48)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # generate backbone features
        conv1, conv2, conv3, conv4, conv5 = self.backbone(input)

        # Semantic Knowledge Compression(SKC) unit, kernel_conv4 (k4) and kernel_conv3 (k3)
        kernel_conv4 = self.pool(self.conv5_conv4(conv5))  # 96*5*5
        kernel_conv3 = self.pool(self.conv5_conv3(conv5))  # 32*5*5

        # conv34 is f_dsmm
        conv34 = self.dsmm(conv4, kernel_conv4, conv3, kernel_conv3)
        # conv12 is f_esam
        edge1, edge2, conv12 = self.esam(conv1, conv2)

        s12, s34, s5 = self.prediction_decoder(conv5, conv34, conv12)

        s5_up = self.upsample8(s5)
        s34_up = self.upsample2(s34)

        return s12, s34_up, s5_up, self.sigmoid(s12), self.sigmoid(s34_up), self.sigmoid(s5_up), edge1, edge2

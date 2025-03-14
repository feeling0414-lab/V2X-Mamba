import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from v2xvit.models.sub_modules import models_mamba

#single LayerVim
# class Vimbackbone(nn.Module):
#     def __init__(self, grid_size):
#         super(Vimbackbone, self).__init__()
#         self.VIM_model = models_mamba.vim_small(grid_size=grid_size)
#         self.downsample =nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=384, kernel_size=3,padding=1),
#             nn.BatchNorm2d(num_features=384),
#             nn.ReLU()
#         )
#
#     def forward(self, data_dict):
#         x = data_dict['spatial_features']
#         x = self.VIM_model(x)
#         x = self.downsample(x)
#         data_dict['spatial_features_2d'] = x
#         return data_dict

#multiScale Vim

#非级联式金字塔结构
# class Vimbackbone(nn.Module):
#     def __init__(self, grid_size):
#         super(Vimbackbone, self).__init__()
#         self.Vim_model1 = models_mamba.vim_1(grid_size=grid_size)
#         self.Vim_model2 = models_mamba.vim_2(grid_size=grid_size)
#         self.Vim_model3 = models_mamba.vim_3(grid_size=grid_size)
#         self.channal_change1 =nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1),
#             nn.BatchNorm2d(num_features=128),
#             nn.ReLU()
#         )
#         self.channal_change2 =nn.Sequential(
#             nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3,padding=1),
#             nn.BatchNorm2d(num_features=256),
#             nn.ReLU()
#         )
#
#         self.upsample1 = nn.Sequential(
#             nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#         )
#         self.upsample2 = nn.Sequential(
#             nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#         )
#
#     def forward(self, data_dict):
#         x = data_dict['spatial_features'] #x[64,192,704]
#         x = self.channal_change1(x)   # x: [128,192,704]
#         x1 = self.Vim_model1(x)
#         x2 = self.Vim_model2(x)
#         x3 = self.Vim_model3(x)
#
#         x = torch.cat((self.upsample1(x1),self.upsample2(x2)), dim=1)
#         x = torch.cat((x, x3), dim=1)
#         x = self.channal_change2(x)
#         data_dict['spatial_features_2d'] = x
#         return data_dict

# #dim不变金字塔结构
# class Vimbackbone(nn.Module):
#     def __init__(self, grid_size):
#         super(Vimbackbone, self).__init__()
#         self.embed_dim = 128
#         self.Vim_model_1 = models_mamba.vim_connect_1(grid_size=grid_size)
#         self.Vim_model_2 = models_mamba.vim_connect_2(grid_size=[int(grid_size[0] / 4),int(grid_size[1] / 4)])
#         self.Vim_model_3 = models_mamba.vim_connect_2(grid_size=[int(grid_size[0] / 8),int(grid_size[1] / 8)])
#         #self.Vim_model_4 = models_mamba.vim_connect_2(grid_size=[int(grid_size[0] /16), int(grid_size[1] /16)])
#         self.channal_change1 =nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=self.embed_dim, kernel_size=3,padding=1),
#             nn.BatchNorm2d(num_features=self.embed_dim,eps=1e-3,momentum=0.1),
#             nn.ReLU()
#         )
#         self.channal_change2 =nn.Sequential(
#             nn.Conv2d(in_channels=self.embed_dim*3, out_channels=256, kernel_size=3,padding=1),
#             nn.BatchNorm2d(num_features=256, eps=1e-3,momentum=0.01),
#             nn.ReLU()
#         )
#
#         self.upsample1 = nn.Sequential(
#             nn.ConvTranspose2d(self.embed_dim,self.embed_dim, kernel_size=2, stride=2),
#             nn.BatchNorm2d(self.embed_dim, eps=1e-3,momentum=0.01),
#             nn.ReLU(),
#         )
#         self.upsample2 = nn.Sequential(
#             nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=4, stride=4),
#             nn.BatchNorm2d(self.embed_dim, eps=1e-3,momentum=0.01),
#             nn.ReLU(),
#         )
#         # self.upsample3 = nn.Sequential(
#         #     nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=8, stride=8),
#         #     nn.BatchNorm2d(self.embed_dim, eps=1e-3,momentum=0.01),
#         #     nn.ReLU(),
#         # )
#
#     def forward(self, data_dict):
#         input_x = data_dict['spatial_features'] #x[64,192,704]
#         input_x = self.channal_change1(input_x)   # x: [128,192,704]
#         x1 = self.Vim_model_1(input_x)        # x1: [128,48,176] 4倍下采样
#         x2 = self.Vim_model_2(x1)       # x2: [128,24,88] 8倍下采样
#         x3 = self.Vim_model_3(x2)       # x3: [128,12,44] 16倍下采样
#         #x4 = self.Vim_model_4(x3)       # x: [128,6,22] 针对大尺度车辆32倍下采样
#
#         #x = torch.cat([self.upsample3(x4), self.upsample2(x3), self.upsample1(x2),x1], dim=1)
#         x = torch.cat([self.upsample2(x3), self.upsample1(x2), x1], dim=1)
#         #x = torch.cat([x1,self.upsample3(x4),self.upsample1(x2),self.upsample2(x3)], dim=1)
#         #x = torch.cat([x1,self.upsample1(x2),self.upsample2(x3)], dim=1)
#         x = self.channal_change2(x)
#         data_dict['spatial_features_2d'] = x
#         return data_dict


# #变结构金字塔结构
# class Vimbackbone(nn.Module):
#     def __init__(self, grid_size):
#         super(Vimbackbone, self).__init__()
#         self.embed_dim=128
#         self.Vim_model_1 = models_mamba.vim_connect_1(grid_size=grid_size)
#         self.Vim_model_2 = models_mamba.vim_connect_2(grid_size=[int(grid_size[0] / 4),int(grid_size[1] / 4)])
#         self.Vim_model_3 = models_mamba.vim_connect_2(grid_size=[int(grid_size[0] / 8),int(grid_size[1] / 8)])
#         self.Vim_model_4 = models_mamba.vim_connect_2(grid_size=[int(grid_size[0] / 16), int(grid_size[1] / 16)])
#
#         self.channal_change1 =nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=self.embed_dim, kernel_size=3,padding=1,bias=False),
#             nn.BatchNorm2d(self.embed_dim, eps=1e-3,momentum=0.01),
#             nn.ReLU()
#         )
#
#         self.channal_change =nn.Sequential(
#             nn.Conv2d(in_channels=self.embed_dim*2, out_channels=256, kernel_size=3,padding=1,bias=False),
#             nn.BatchNorm2d(256, eps=1e-3,momentum=0.01),
#             nn.ReLU()
#         )
#
#         self.upsample1 = nn.Sequential(
#             nn.ConvTranspose2d(self.embed_dim,self.embed_dim, kernel_size=2, stride=2,bias=False),
#             nn.BatchNorm2d(self.embed_dim, eps=1e-3,momentum=0.01),
#             nn.ReLU(),
#         )
#         self.upsample2 = nn.Sequential(
#             nn.ConvTranspose2d(self.embed_dim*2,self.embed_dim, kernel_size=2, stride=2,bias=False),
#             nn.BatchNorm2d(self.embed_dim, eps=1e-3,momentum=0.01),
#             nn.ReLU(),
#         )

#
#     def forward(self, data_dict):
#         input_x = data_dict['spatial_features'] #x[64,192,704]
#         input_x = self.channal_change1(input_x)
#         x1 = self.Vim_model_1(input_x)        # x1: [128,48,176] 4倍下采样
#         x2 = self.Vim_model_2(x1)       # x2: [128,24,88] 8倍下采样
#         x3 = self.Vim_model_3(x2)       # x3: [128,12,44] 16倍下采样
#         x4 = self.Vim_model_4(x3)       # x: [128,6,22] 针对大尺度车辆32倍下采样
#
#
#         x3 = torch.cat([self.upsample1(x4), x3],dim=1)
#         x2 = torch.cat([self.upsample2(x3), x2], dim=1)
#         x1 = torch.cat([self.upsample2(x2), x1], dim=1)
#
#         x = self.channal_change(x1)
#         data_dict['spatial_features_2d'] = x
#         return data_dict
#



# 增加residual结构的金字塔结构
class Vimbackbone(nn.Module):
    def __init__(self, grid_size):
        super(Vimbackbone, self).__init__()
        self.embed_dim = 128
        self.Vim_model_1 = models_mamba.vim_connect_1(grid_size=grid_size)
        self.Vim_model_2 = models_mamba.vim_connect_2(grid_size=[int(grid_size[0] / 4), int(grid_size[1] / 4)])
        self.Vim_model_3 = models_mamba.vim_connect_2(grid_size=[int(grid_size[0] / 8), int(grid_size[1] / 8)])
        self.Vim_model_4 = models_mamba.vim_connect_2(grid_size=[int(grid_size[0] / 16), int(grid_size[1] / 16)])

        self.channal_change1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=self.embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dim, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        self.channal_change = nn.Sequential(
            nn.Conv2d(in_channels=self.embed_dim * 2, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(self.embed_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim * 2, self.embed_dim, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(self.embed_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.resBlock1 = resBlock(self.embed_dim)
        self.resBlock2 = resBlock(self.embed_dim*2)
    def forward(self, data_dict):
        input_x = data_dict['spatial_features']  # x[64,192,704]
        input_x = self.channal_change1(input_x)
        x1 = self.Vim_model_1(input_x)  # x1: [128,48,176] 4倍下采样
        x2 = self.Vim_model_2(x1)  # x2: [128,24,88] 8倍下采样
        x3 = self.Vim_model_3(x2)  # x3: [128,12,44] 16倍下采样
        x4 = self.Vim_model_4(x3)  # x: [128,6,22] 针对大尺度车辆32倍下采样

        x4 = self.resBlock1(x4)
        x3 = torch.cat([self.upsample1(x4), x3], dim=1)
        x3 = self.resBlock2(x3)
        x2 = torch.cat([self.upsample2(x3), x2], dim=1)
        x2 = self.resBlock2(x2)
        x1 = torch.cat([self.upsample2(x2), x1], dim=1)
        x1 = self.resBlock2(x1)
        x = self.channal_change(x1)
        data_dict['spatial_features_2d'] = x
        return data_dict




class resBlock(nn.Module):
    def __init__(self,feature_dim):
        super(resBlock,self).__init__()
        self.embed_dim = feature_dim
        self.residual = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.ReLU = nn.ReLU()
    def forward(self, feature):
        feature_mid = self.residual(feature)
        feature_out = feature+feature_mid
        feature_out = self.ReLU(feature_out)
        return feature_out


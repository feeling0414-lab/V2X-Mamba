# import torch
# from torchvision.ops import roi_pool
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
#
# # 假设我们有以下输入：
# input_feature_map = torch.randn(10, 256, 32, 32, dtype=torch.float32)  # [N, C, H, W]，N是批量大小，C是通道数，H和W是特征图的高度和宽度
# rois = torch.tensor([[0, 1, 1, 10, 10],  # 对于每个ROI，前一个数字是batch索引，后四个是左上角和右下角坐标
#                      [1, 30, 30, 100, 100]], dtype=torch.float32)  # [K, 5]，K是ROI的数量,这里K=2
#
# # 定义输出尺寸（例如7x7）
# output_size = (7, 7)
# # 定义空间比例因子，通常是在特征图与原始图像之间做归一化处理
# spatial_scale = 1.0 / 16  # 假设特征图是原图经过16倍下采样得到的
#
# # 应用ROI Pooling
# pooled_features = roi_pool(input_feature_map, rois, output_size, spatial_scale)
# print(pooled_features.shape)
#
# fig = plt.figure(dpi=400)
# plt.imshow(input_feature_map[0][0])
#
# for i in range(2):
#     # 创建一个矩形框
#     a = pooled_features[i][0]
#     bbox = patches.Rectangle((pooled_features[i][0]), 10, 10, linewidth=2, edgecolor='r', facecolor='none')
#     # 将矩形框添加到图像中
#     plt.gca().add_patch(bbox)
# plt.show()
# print()

import torch


def generate_rois(proposals, image_shape):
    """
    生成Region of Interest (RoI)。

    参数:
    proposals (Tensor): 提议框，形状为(N, 4)，其中N是提议框的数量。
    image_shape (tuple): 图像的尺寸，形式为(height, width)。

    返回:
    rois (Tensor): 归一化的Region of Interest，形状为(N, 5)，其中包含[batch_index, x1, y1, x2, y2]。
    """
    batch_size = proposals.size(0)
    # 初始化输出的RoI张量
    rois = torch.zeros((batch_size, 5), dtype=torch.float32)
    # 填充batch_index
    rois[:, 0] = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).squeeze(1)
    # 获取图像的宽度和高度
    height, width = image_shape
    # 将提议框的坐标归一化到0到1之间
    x1, y1, x2, y2 = proposals.chunk(4, dim=1)
    rois[:, 1] = x1 / width
    rois[:, 2] = y1 / height
    rois[:, 3] = x2 / width
    rois[:, 4] = y2 / height
    return rois


# 示例使用
proposals = torch.tensor([[50, 20, 100, 70]], dtype=torch.float32)
image_shape = (100, 200)  # 假设图像尺寸为100x200
rois = generate_rois(proposals, image_shape)
print(rois)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
#show attention_map
# class ScaledDotProductAttention(nn.Module):
#     """
#     Scaled Dot-Product Attention proposed in "Attention Is All You Need"
#     Compute the dot products of the query with all keys, divide each by sqrt(dim),
#     and apply a softmax function to obtain the weights on the values
#     Args: dim, mask
#         dim (int): dimention of attention
#         mask (torch.Tensor): tensor containing indices to be masked
#     Inputs: query, key, value, mask
#         - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
#         - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
#         - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
#         - **mask** (-): tensor containing indices to be masked
#     Returns: context, attn
#         - **context**: tensor containing the context vector from attention mechanism.
#         - **attn**: tensor containing the attention (alignment) from the encoder outputs.
#     """
#
#     def __init__(self, dim):
#         super(ScaledDotProductAttention, self).__init__()
#         self.sqrt_dim = np.sqrt(dim)
#
#     def forward(self, query, key, value):
#         score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
#         attn = F.softmax(score, -1)
#         context = torch.bmm(attn, value)
#         return attn,context
#
#         # return context
#
#
# class AttFusion(nn.Module):
#     def __init__(self, feature_dim):
#         super(AttFusion, self).__init__()
#         self.att = ScaledDotProductAttention(feature_dim)
#
#     def forward(self, x, record_len):
#         split_x = self.regroup(x, record_len)
#         batch_size = len(record_len)
#         C, W, H = split_x[0].shape[1:]
#         out = []
#         att_out = []
#         for xx in split_x:
#             cav_num = xx.shape[0]
#             xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
#             att,h = self.att(xx, xx, xx)
#             h = h.permute(1, 2, 0)
#             h = h.view(cav_num, C, W, H)
#             h = h[0, ...].unsqueeze(0)
#             att = att.permute(2, 1, 0).view(cav_num, cav_num, W, H)
#             out.append(h)
#             att_out.append(att)
#         return torch.cat(att_out,dim=0),torch.cat(out, dim=0)
#
#     def regroup(self, x, record_len):
#         cum_sum_len = torch.cumsum(record_len, dim=0)
#         split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
#         return split_x



class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context

class AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)
        batch_size = len(record_len)
        C, W, H = split_x[0].shape[1:]
        out = []
        for xx in split_x:
            cav_num = xx.shape[0]
            xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
            h = self.att(xx, xx, xx)
            h = h.permute(1, 2, 0).view(cav_num, C, W, H)[0, ...].unsqueeze(0)
            out.append(h)
        return torch.cat(out, dim=0)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x



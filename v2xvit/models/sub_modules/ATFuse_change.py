import math

import torch
from torch import einsum, nn
import numpy as np
from functools import partial
import torch.nn.functional as F
from torch.nn import Softmin
torch.autograd.set_detect_anomaly(True)


class Mlp(nn.Module):
    """Feed-forward network (FFN, a.k.a.

    MLP) class.
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.2,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """foward function"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#卷积相对位置编码的因子化注意力
class FactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class."""

    def __init__(
            self,
            dim,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.2,
            shared_crpe=None,
    ):
        super().__init__()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe
        self.sqrt_dim = np.sqrt(dim)
    def forward(self, q, k, v, minus=True):
        N, C = q.shape

        # Generate Q, K, V.
        q = self.q(q).reshape(N, C)
        k = self.k(k).reshape(N, C)
        v = self.v(v).reshape(N, C)
        # Factorized attention.
        use_efficient = minus
        score = torch.mm(q,k.transpose(1,0))/ self.sqrt_dim
        attn = F.softmax(score, -1)
        factor_att = torch.mm(attn,v)

        # Merge and reshape.
        if use_efficient:
            x = v - factor_att
        else:
            x = factor_att
        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MHCABlock(nn.Module):
    """Multi-Head Convolutional self-Attention block."""
    def __init__(
            self,
            dim,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            shared_cpe=None,
            shared_crpe=None,
    ):
        super().__init__()

        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=dim)

        self.norm2 = norm_layer(dim)

    #minus True
    def forward(self, q, k, v, minus=True):
        """foward function"""
        """foward function"""
        x = q + self.factoratt_crpe(q, k, v, minus)
        cur = self.norm2(x)
        x = x + self.mlp(cur)
        return x

class ATF(nn.Module):
    def __init__(self, feature_dim):
        super(ATF, self).__init__()
        self.MHCA = MHCABlock(dim = feature_dim)


    def forward(self, ego,other):
        #input sequence Q,K,V
        DIIM_ego     = self.MHCA(other,ego,ego,  minus =True) # as q
        ACIIM_other  = self.MHCA(DIIM_ego, other, other, minus = False)
        ACIIM_other2 = self.MHCA(ACIIM_other, ego, ego, minus = False)
        a = ACIIM_other2 + DIIM_ego

        return a

class FusionNet(nn.Module):
    def __init__(self, feature_dim):

        super(FusionNet, self).__init__()
        self.ATF = ATF(feature_dim=feature_dim)

        self.BN = nn.Sequential(nn.Conv2d(feature_dim,feature_dim, kernel_size=1),
                                nn.BatchNorm2d(feature_dim, eps=1e-3, momentum=0.01),
                                nn.ReLU())


    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)
        batch_size = len(record_len)
        C, W, H = split_x[0].shape[1:]
        out = []

        for xx in split_x:
            cav_num = xx.shape[0]
            xx = xx.view(cav_num,C, -1)
            ego = xx[0, :, :]

            if cav_num > 1:
                for i in range(cav_num-1):
                    ego  = ego.transpose(1, 0)
                    other = xx[i+1,:,:].transpose(1, 0)
                    ego_att = self.ATF(ego, other).transpose(1, 0)
                    ego = ego_att
                h = ego.view(C, W, H).unsqueeze(0)
                out.append(h)
            else:
                h = ego.view(C, W, H).unsqueeze(0)
                out.append(h)
        a = torch.cat(out, dim=0)
        a = self.BN(a)
        return a

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
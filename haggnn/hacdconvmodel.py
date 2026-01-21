import torch.nn as nn
from torch_geometric.nn import MLP, fps, global_max_pool, global_mean_pool, radius
from hacdconvlayer1 import CDConv, MaxPooling
from typing import Type, Any, Callable, Union, List, Optional
import torch
from sklearn.preprocessing import normalize
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_max, scatter_mean
import numpy as np


def orientation(pos):
    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    u = normalize(X=pos[1:, :] - pos[:-1, :], norm='l2', axis=1)
    u1 = u[1:, :]
    u2 = u[:-1, :]
    b = normalize(X=u2 - u1, norm='l2', axis=1)
    n = normalize(X=np.cross(u2, u1), norm='l2', axis=1)
    o = normalize(X=np.cross(b, n), norm='l2', axis=1)
    ori = np.stack([b, n, o], axis=1)
    # Convert the numpy array to a Tensor
    ori_tensor = torch.tensor(np.concatenate([np.expand_dims(ori[0], 0), ori, np.expand_dims(ori[-1], 0)], axis=0),
                              dtype=torch.float32)
    print(f"orientation shape:{ori_tensor.shape}")
    return ori_tensor


def global_mean_pool_no_batch(x):

    return torch.mean(x, dim=0)


class Linear(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:
        super(Linear, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        module.append(nn.Linear(in_channels, out_channels, bias=bias))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x)


class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 batch_norm: bool,
                 dropout: float = 0.0,
                 bias: bool = True,
                 leakyrelu_negative_slope: float = 0.2,
                 momentum: float = 0.2) -> nn.Module:
        super(MLP, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias=bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias=bias))
        if batch_norm:
            if mid_channels is None:
                module.append(nn.BatchNorm1d(out_channels, momentum=momentum))
            else:
                module.append(nn.BatchNorm1d(mid_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias=bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)


class BasicBlock(nn.Module):
    def __init__(self,
                 r: float,
                 l: float,
                 kernel_channels: list[int],
                 in_channels: int,
                 out_channels: int,
                 base_width: float = 64.0,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:

        super(BasicBlock, self).__init__()

        if in_channels != out_channels:
            self.identity = Linear(in_channels=in_channels,
                                   out_channels=out_channels,
                                   batch_norm=batch_norm,
                                   dropout=dropout,
                                   bias=bias,
                                   leakyrelu_negative_slope=leakyrelu_negative_slope,
                                   momentum=momentum)
        else:
            self.identity = nn.Sequential()

        width = int(out_channels * (base_width / 64.))
        self.input = MLP(in_channels=in_channels,
                         mid_channels=None,
                         out_channels=width,
                         batch_norm=batch_norm,
                         dropout=dropout,
                         bias=bias,
                         leakyrelu_negative_slope=leakyrelu_negative_slope,
                         momentum=momentum)
        self.conv = CDConv(r=r, l=l, kernel_channels=kernel_channels, in_channels=width, out_channels=width)

        self.output = Linear(in_channels=width,
                             out_channels=out_channels,
                             batch_norm=batch_norm,
                             dropout=dropout,
                             bias=bias,
                             leakyrelu_negative_slope=leakyrelu_negative_slope,
                             momentum=momentum)

    def forward(self, x, pos, seq, ori, batch, water_shell_layers):
        identity = self.identity(x)
        x = self.conv(x, pos, seq, ori, batch, water_shell_layers=water_shell_layers)
        out = self.output(x) + identity
        return out


class AvgPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, pos, seq, ori, batch, water_shells):
        idx = torch.div(seq.squeeze(1), 2, rounding_mode='floor')
        idx = torch.cat([idx, idx[-1].view((1,))])
        idx = (idx[0:-1] != idx[1:]).to(torch.float32)
        idx = torch.cumsum(idx, dim=0) - idx
        idx = idx.to(torch.int64)
        x = scatter_mean(src=x, index=idx, dim=0)
        pos = scatter_mean(src=pos, index=idx, dim=0)
        seq = scatter_max(src=torch.div(seq, 2, rounding_mode='floor'), index=idx, dim=0)[0]
        ori = scatter_mean(src=ori, index=idx, dim=0)
        ori = torch.nn.functional.normalize(ori, 2, -1)
        water_shells = scatter_max(src=water_shells.float(), index=idx, dim=0)[0].long()
        batch = scatter_max(src=batch, index=idx, dim=0)[0]
        return x, pos, seq, ori, batch, water_shells


class HACDConvModel(nn.Module):
    def __init__(self,
                 geometric_radii: List[float],
                 sequential_kernel_size: float,
                 kernel_channels: List[int],
                 channels: List[int],
                 base_width: float = 64.0,
                 embedding_dim: int = 27,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 water_shell_data=None,
                 num_classes: int = 384
                 ) -> None:
        super().__init__()

        assert (len(geometric_radii) == len(
            channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

        self.embedding = torch.nn.Embedding(num_embeddings=27, embedding_dim=embedding_dim)
        self.local_mean_pool = AvgPooling()

        layers = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r=radius,
                                     l=sequential_kernel_size,
                                     kernel_channels=kernel_channels,
                                     in_channels=in_channels,
                                     out_channels=channels[i],
                                     base_width=base_width,
                                     batch_norm=batch_norm,
                                     dropout=dropout,
                                     bias=bias))
            layers.append(BasicBlock(r=radius,
                                     l=sequential_kernel_size,
                                     kernel_channels=kernel_channels,
                                     in_channels=channels[i],
                                     out_channels=channels[i],
                                     base_width=base_width,
                                     batch_norm=batch_norm,
                                     dropout=dropout,
                                     bias=bias))
            in_channels = channels[i]

        self.layers = nn.Sequential(*layers)

    def forward(self, data):
        x, pos, seq, ori, batch, water_shell_layers = data.x, data.pos, data.seq, data.ori, data.batch, data.water_shell_layers
        for i, layer in enumerate(self.layers):
            x = layer(x, pos, seq, ori, batch, water_shell_layers)
            if i == len(self.layers) - 1:
                x = global_mean_pool(x, batch)
            elif i % 2 == 1:
                x, pos, seq, ori, batch = self.local_mean_pool(x, pos, seq, ori, batch)
        out = x
        return out
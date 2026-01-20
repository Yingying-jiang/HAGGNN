import torch_scatter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from functools import partial
import torch.nn.functional as F
import torch.nn as nn
import torch
from jaxtyping import Bool, Float, Int64, jaxtyped
from typing import Callable, Optional, Union, Tuple
from torchtyping import TensorType, patch_typeguard
from torch_scatter import scatter


def get_nonlinearity(nonlinearity: Optional[str] = None, slope: float = 1e-2, return_functional: bool = False) -> Any:
    if isinstance(nonlinearity, str):
        nonlinearity = nonlinearity.lower().strip()
    elif isinstance(nonlinearity, torch.Tensor):
        raise ValueError(f"Expected a string for nonlinearity, but got a Tensor: {nonlinearity}")

    if nonlinearity == "relu":
        return F.relu if return_functional else nn.ReLU()
    elif nonlinearity == "leakyrelu":
        return partial(F.leaky_relu, negative_slope=slope) if return_functional else nn.LeakyReLU(negative_slope=slope)
    elif nonlinearity == "selu":
        return partial(F.selu) if return_functional else nn.SELU()
    elif nonlinearity == "silu":
        return partial(F.silu) if return_functional else nn.SiLU()
    elif nonlinearity == "sigmoid":
        return torch.sigmoid if return_functional else nn.Sigmoid()
    elif nonlinearity is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"The nonlinearity {nonlinearity} is currently not implemented.")


def scalarize(
    vector_rep: TensorType["batch_num_entities", 3, 3],
    edge_index: TensorType[2, "batch_num_edges"],
    frames: TensorType["batch_num_edges", 3, 3],
    node_inputs: bool,
    enable_e3_equivariance: bool,
    dim_size: int,
    node_mask: Optional[TensorType["batch_num_nodes"]] = None
) -> TensorType["effective_batch_num_entities", 9]:
    row, col = edge_index[0], edge_index[1]
    vector_rep_i = vector_rep[row] if node_inputs else vector_rep
    if vector_rep_i.ndim == 2:
        vector_rep_i = vector_rep_i.unsqueeze(-1)
    elif vector_rep_i.ndim == 3:
        vector_rep_i = vector_rep_i.transpose(-1, -2)

    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]
        local_scalar_rep_i = torch.zeros((edge_index.shape[1], 3, 3), device=edge_index.device)
        local_scalar_rep_i[edge_mask] = torch.matmul(
            frames[edge_mask], vector_rep_i[edge_mask]
        )
        local_scalar_rep_i = local_scalar_rep_i.transpose(-1, -2)
    else:
        local_scalar_rep_i = torch.matmul(frames, vector_rep_i).transpose(-1, -2)

    if enable_e3_equivariance:
        local_scalar_rep_i_copy = local_scalar_rep_i.clone()
        local_scalar_rep_i_copy[:, :, 1] = torch.abs(local_scalar_rep_i[:, :, 1])
        local_scalar_rep_i = local_scalar_rep_i_copy
    local_scalar_rep_i = local_scalar_rep_i.reshape(vector_rep_i.shape[0], 9)

    if node_inputs:
        return scatter(
            local_scalar_rep_i,
            row,
            dim=0,
            dim_size=dim_size,
            reduce="mean"
        )

    return local_scalar_rep_i

def is_identity(
    nonlinearity: Optional[Union[Callable, nn.Module]] = None
) -> bool:
    return nonlinearity is None or isinstance(nonlinearity, nn.Identity)


def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    keepdim: bool = False,
    sqrt: bool = True,
) -> torch.Tensor:
    norm = torch.sum(x**2, dim=dim, keepdim=keepdim)
    if sqrt:
        norm = torch.sqrt(norm + eps)
    return norm + eps

def vectorize(
    self,
    scalar_rep: Float[torch.Tensor, "batch_num_entities merged_scalar_dim"],
    vector_hidden_rep: Float[torch.Tensor, "batch_num_entities 3 n"],
) -> Float[torch.Tensor, "batch_num_entities o 3"]:
    vector_rep = self.vector_up(vector_hidden_rep)
    vector_rep = vector_rep.transpose(-1, -2)

    if self.vector_gate:
        gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
        vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
    elif not is_identity(self.vector_nonlinearity):
        vector_rep = vector_rep * self.vector_nonlinearity(
            safe_norm(vector_rep, dim=-1, keepdim=True)
        )

    return vector_rep


class ScalarVector(tuple):
    """
    From https://github.com/sarpaykent/GBPNet
    """
    def __new__(cls, scalar, vector):
        return tuple.__new__(cls, (scalar, vector))

    def __getnewargs__(self):
        return self.scalar, self.vector

    @property
    def scalar(self):
        return self[0]

    @property
    def vector(self):
        return self[1]

    def __add__(self, other):
        if isinstance(other, tuple):
            scalar_other = other[0]
            vector_other = other[1]
        else:
            scalar_other = other.scalar
            vector_other = other.vector

        return ScalarVector(self.scalar + scalar_other, self.vector + vector_other)

    def __mul__(self, other):
        if isinstance(other, tuple):
            other = ScalarVector(other[0], other[1])

        if isinstance(other, ScalarVector):
            return ScalarVector(self.scalar * other.scalar, self.vector * other.vector)
        else:
            return ScalarVector(self.scalar * other, self.vector * other)

    def concat(self, others, dim=-1):
        dim %= len(self.scalar.shape)
        s_args, v_args = list(zip(*(self, *others)))
        return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

    def flatten(self):
        flat_vector = torch.reshape(self.vector, self.vector.shape[:-2] + (3 * self.vector.shape[-2],))
        return torch.cat((self.scalar, flat_vector), dim=-1)

    @staticmethod
    def recover(x, vector_dim):
        v = torch.reshape(x[..., -3 * vector_dim:], x.shape[:-1] + (vector_dim, 3))
        s = x[..., : -3 * vector_dim]
        return ScalarVector(s, v)

    def vs(self):
        return self.scalar, self.vector

    def idx(self, idx):
        return ScalarVector(self.scalar[idx], self.vector[idx])

    def repeat(self, n, c=1, y=1):
        return ScalarVector(self.scalar.repeat(n, c), self.vector.repeat(n, y, c))

    def clone(self):
        return ScalarVector(self.scalar.clone(), self.vector.clone())

    def mask(self, node_mask: TensorType["num_nodes"]):
        return ScalarVector(
            self.scalar * node_mask[:, None],
            self.vector * node_mask[:, None, None]
        )

    def __setitem__(self, key, value):
        self.scalar[key] = value.scalar
        self.vector[key] = value.vector

    def __repr__(self):
        return f"ScalarVector({self.scalar}, {self.vector})"


class GCPLayerNorm(nn.Module):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, dims: ScalarVector, eps: float = 1e-8):
        super(GCPLayerNorm, self).__init__()
        self.scalar_dims, self.vector_dims = dims
        self.scalar_norm = nn.LayerNorm(self.scalar_dims)
        self.eps = eps

    @staticmethod
    def norm_vector(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        vector_norm = torch.clamp(torch.sum(torch.square(v), dim=-1, keepdim=True), min=eps)
        vector_norm = torch.sqrt(torch.mean(vector_norm, dim=-2, keepdim=True))
        return v / vector_norm

    def forward(self, x: Union[torch.Tensor, ScalarVector]):
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (x.scalar.shape[0] == 0 or x.vector.shape[0] == 0):
            return x
        elif not self.vector_dims:
            return (
                self.scalar_norm(x[0])
                if isinstance(x, ScalarVector)
                else self.scalar_norm(x)
            )
        s, v = x
        return ScalarVector(self.scalar_norm(s), self.norm_vector(v, eps=self.eps))
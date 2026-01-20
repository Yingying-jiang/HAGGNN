import torch
from torch import Tensor
from torch import nn
from layer import (scalarize, vectorize, ScalarVector, is_identity, safe_norm)
from torchtyping import TensorType
from typing import Any, Optional, Tuple, Union
import dgl
from dgl.nn import Set2Set
import torch.nn.functional as F



class GCP(nn.Module):
    def __init__(
            self,
            input_dims: ScalarVector,
            output_dims: ScalarVector,
            scalar_gate: int = 1,
            vector_gate: bool = True,
            frame_gate: bool = False,
            sigma_frame_gate: bool = False,
            bottleneck: int = 1,
            vector_residual: bool = False,
            vector_frame_residual: bool = False,
            ablate_frame_updates: bool = False,
            ablate_scalars: bool = False,
            ablate_vectors: bool = False,
            enable_e3_equivariance: bool = False,
            scalarization_vectorization_output_dim: int = 3,
            water_shell_layers: Optional[Tensor] = None,
            shell_log_weights: Optional[Tensor] = None,
            **kwargs
    ):
        super(GCP, self).__init__()
        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.scalar_nonlinearity = torch.nn.ReLU()
        self.vector_nonlinearity = torch.nn.Sigmoid()
        self.scalar_gate, self.vector_gate, self.frame_gate, self.sigma_frame_gate = (
            scalar_gate, vector_gate, frame_gate, sigma_frame_gate
        )
        self.vector_residual, self.vector_frame_residual = vector_residual, vector_frame_residual
        self.ablate_frame_updates = ablate_frame_updates
        self.ablate_scalars, self.ablate_vectors = ablate_scalars, ablate_vectors
        self.enable_e3_equivariance = enable_e3_equivariance
        self.water_shell_layers = water_shell_layers
        self.shell_log_weights = nn.Parameter(torch.randn(3))
        self.register_buffer('water_shell_layers_j', water_shell_layers)
        scalar_gate_value = self.scalar_gate

        if scalar_gate_value > 0:
           print("scalar_gate is positive")
        else:
            print("scalar_gate is non-positive")
        if water_shell_layers is not None:
            self.register_buffer('water_shell_layers_j', water_shell_layers)
        else:
            self.water_shell_layers_j = None

        self.scalar_gate = max(1, scalar_gate)
        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.scalar_output_dim)

        if self.vector_input_dim:
            assert (
                self.vector_input_dim % bottleneck == 0
            ), f"Input channel of vector ({self.vector_input_dim}) must be divisible with bottleneck factor ({bottleneck})"

            self.hidden_dim = self.vector_input_dim // bottleneck if bottleneck > 1 else max(self.vector_input_dim,
                                                                                             self.vector_output_dim)

            self.vector_down = nn.Linear(self.vector_input_dim, self.hidden_dim, bias=False)
            self.scalar_out = nn.Linear(self.hidden_dim + self.scalar_input_dim, self.scalar_output_dim)

            if self.vector_output_dim:
                self.vector_up = nn.Linear(self.hidden_dim, self.vector_output_dim, bias=False)
                if self.vector_gate:
                    self.vector_out_scale = nn.Linear(self.scalar_output_dim, self.vector_output_dim)

            if not self.ablate_frame_updates:
                vector_down_frames_input_dim = self.hidden_dim if not self.vector_output_dim else self.vector_output_dim
                self.vector_down_frames = nn.Linear(vector_down_frames_input_dim,
                                                    scalarization_vectorization_output_dim, bias=False)
                self.scalar_out_frames = nn.Linear(
                    self.scalar_output_dim + scalarization_vectorization_output_dim * 3, self.scalar_output_dim)

                if self.vector_output_dim and self.sigma_frame_gate:
                    self.vector_out_scale_sigma_frames = nn.Linear(self.scalar_output_dim, self.vector_output_dim)
                elif self.vector_output_dim and self.frame_gate:
                    self.vector_out_scale_frames = nn.Linear(
                        self.scalar_output_dim, scalarization_vectorization_output_dim * 3)
                    self.vector_up_frames = nn.Linear(
                        scalarization_vectorization_output_dim, self.vector_output_dim, bias=False)
        else:
            self.scalar_out = nn.Linear(self.scalar_input_dim, self.scalar_output_dim)

    def process_vector(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"],
        v_pre: TensorType["batch_num_entities", 3, "m"],
        vector_hidden_rep: TensorType["batch_num_entities", 3, "n"]
    ) -> TensorType["batch_num_entities", "o", 3]:
        vector_rep = self.vector_up(vector_hidden_rep)
        if self.vector_residual:
            vector_rep = vector_rep + v_pre
        vector_rep = vector_rep.transpose(-1, -2)
        if self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    def create_zero_vector(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"]
    ) -> TensorType["batch_num_entities", "o", 3]:
        return torch.zeros(scalar_rep.shape[0], self.vector_output_dim, 3, device=scalar_rep.device)

    def process_vector_frames(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"],
        v_pre: TensorType["batch_num_entities", 3, "o"],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_inputs: bool,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> TensorType["batch_num_entities", "p", 3]:
        vector_rep = v_pre.transpose(-1, -2)
        if self.sigma_frame_gate:
            gate = self.vector_out_scale_sigma_frames(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif self.frame_gate:
            gate = self.vector_out_scale_frames(self.vector_nonlinearity(scalar_rep))
            gate_vector = vectorize(
                gate,
                edge_index,
                frames,
                node_inputs=node_inputs,
                dim_size=scalar_rep.shape[0],
                node_mask=node_mask
            )
            gate_vector_rep = self.vector_up_frames(gate_vector.transpose(-1, -2)).transpose(-1, -2)
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(gate_vector_rep, dim=-1, keepdim=True))
            if self.vector_frame_residual:
                vector_rep = vector_rep + v_pre.transpose(-1, -2)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep


    def forward(
        self,
        s_maybe_v: Union[
        Tuple[
            TensorType["batch_num_entities", "scalar_dim"],
            TensorType["batch_num_entities", "m", "vector_dim"]
        ],
        TensorType["batch_num_entities", "merged_scalar_dim"]
    ],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        water_shell_layers: Optional[Tensor] = None,
        node_inputs: bool = True,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
) -> Union[
        Tuple[
        TensorType["batch_num_entities", "new_scalar_dim"],
        TensorType["batch_num_entities", "n", "vector_dim"]
    ],
        TensorType["batch_num_entities", "new_scalar_dim"]
]:
        if water_shell_layers is not None:
            self.water_shell_layers_j = water_shell_layers

        scalar_rep = None
        vector_rep = None

        if isinstance(s_maybe_v, tuple):
            scalar_rep, vector_rep = s_maybe_v
        else:
            scalar_rep = s_maybe_v

        node_out_combined = edge_index[1]

        if self.water_shell_layers_j is not None and scalar_rep is not None:
            try:
                water_shell_layers_j = self.water_shell_layers_j.to(node_out_combined.device)

                shell_j = water_shell_layers_j[node_out_combined]
                index = (shell_j - 1).long()
                index = torch.clamp(index, 0, self.shell_log_weights.size(0) - 1)
                water_weight = self.shell_log_weights[index].unsqueeze(-1)
                aggregated_weight = torch.zeros(
                scalar_rep.size(0), 
                device=water_weight.device,
                dtype=water_weight.dtype
            )
                aggregated_weight.scatter_add_(
                0, 
                node_out_combined, 
                water_weight.squeeze(-1)
            )
                scalar_rep = scalar_rep * aggregated_weight.unsqueeze(1)
                if vector_rep is not None:
                    vector_rep = vector_rep * aggregated_weight.view(-1, 1, 1)
                
            except Exception as e:
                print(f"water shell weight error: {e}")

        if self.vector_input_dim and vector_rep is not None:
            v_pre = vector_rep.transpose(-1, -2)
            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1) if scalar_rep is not None else vector_norm
        else:
            merged = scalar_rep if scalar_rep is not None else torch.zeros_like(s_maybe_v)
        scalar_rep = self.scalar_out(merged) if scalar_rep is not None else None
        if self.vector_input_dim and self.vector_output_dim and vector_rep is not None:
            vector_rep = self.process_vector(scalar_rep, v_pre, vector_hidden_rep)
        if scalar_rep is not None:
            scalar_rep = self.scalar_nonlinearity(scalar_rep)
        if self.vector_output_dim and not self.vector_input_dim and scalar_rep is not None:
            vector_rep = self.create_zero_vector(scalar_rep)
        if self.ablate_frame_updates:
            if self.vector_output_dim and vector_rep is not None:
                return ScalarVector(scalar_rep, vector_rep)
            else:
                return scalar_rep
        if vector_rep is not None:
            v_pre = vector_rep.transpose(-1, -2)
            vector_hidden_rep = self.vector_down_frames(v_pre)
            scalar_hidden_rep = scalarize(
            vector_hidden_rep.transpose(-1, -2),
            edge_index,
            frames,
            node_inputs=node_inputs,
            enable_e3_equivariance=self.enable_e3_equivariance,
            dim_size=vector_hidden_rep.shape[0],
            node_mask=node_mask
        )
            if scalar_rep is not None:
                merged = torch.cat((scalar_rep, scalar_hidden_rep), dim=-1)
                scalar_rep = self.scalar_out_frames(merged)
        if not self.vector_output_dim:
            return self.scalar_nonlinearity(scalar_rep) if scalar_rep is not None else None
        if self.vector_input_dim and self.vector_output_dim and vector_rep is not None:
            vector_rep = self.process_vector_frames(
            scalar_rep,
            v_pre,
            edge_index,
            frames,
            node_inputs=node_inputs,
            node_mask=node_mask
        )
        if scalar_rep is not None:
            scalar_rep = self.scalar_nonlinearity(scalar_rep)
        if self.vector_output_dim and vector_rep is not None:
            return ScalarVector(scalar_rep, vector_rep)
        else:
            return scalar_rep
    

class GCPRegressor(nn.Module):
    def __init__(
            self,
            gcp_config: dict,
            mlp_hidden_dim: int = 256,
            output_dim: int = 1,
            gat_heads: int = 4,
            pooling_type: str = 'hierarchical'
    ):
        super().__init__()
        self.pooling_type = pooling_type
        water_shell_layers = gcp_config.get('water_shell_layers', None)
        shell_log_weights = gcp_config.get('shell_log_weights', None)
        input_dim = gcp_config['scalar_in']
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.output_norm = output_dim
        self.output_norm = nn.BatchNorm1d(gcp_config['scalar_out'])
        self.post_gcp_activation = nn.Softplus()
        self.gcp1 = GCP(
            input_dims=(gcp_config['scalar_in'], gcp_config['vector_in']),
            output_dims=(128, 3),
            hidden_dim=256,
            water_shell_layers=water_shell_layers,
            shell_log_weights=shell_log_weights
        )
        self.gcp2 = GCP(
            input_dims=(128, 3),
            output_dims=(gcp_config['scalar_out'], 3),
            hidden_dim=256,
            water_shell_layers=water_shell_layers,
            shell_log_weights=shell_log_weights
        )
        self.gat = dgl.nn.GATConv(
            in_feats=gcp_config['scalar_out'],
            out_feats=128,
            num_heads=gat_heads,
            feat_drop=0.2,
            attn_drop=0.1
        )
        if pooling_type == 'hierarchical':
            self.pool1 = dgl.nn.AvgPooling()
            self.pool2 = dgl.nn.MaxPooling()
            self.pool3 = dgl.nn.SortPooling(k=20)
            self.pool_output_dim = gcp_config['scalar_out'] + 128 + 20 * gcp_config['scalar_out']
        else:
            self.pool = dgl.nn.Set2Set(input_dim=128, n_iters=6, n_layers=1)
            self.pool_output_dim = 2 * 128
        self.mlp = nn.Sequential(
            nn.Linear(self.pool_output_dim, mlp_hidden_dim*2),
            nn.BatchNorm1d(mlp_hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.6),
            nn.Linear(mlp_hidden_dim*2, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(mlp_hidden_dim // 2, output_dim)
        )
        self.res_conv = nn.Linear(128, gcp_config['scalar_out'])
        self.res_linear = nn.Linear(gcp_config['scalar_out'], 128)

    def forward(self, batched_graph, s_maybe_v, edge_index, frames, water_shell_layers):
        def check_nan(tensor, name):
            if torch.isnan(tensor).any():
                print(f"NaN detected in {name} at forward pass")
                tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
            return tensor
        scalar_in, vector_in = s_maybe_v
        scalar_in = self.input_norm(scalar_in)
        s_maybe_v = (scalar_in, vector_in)
        gcp1_output = self.gcp1(s_maybe_v, edge_index, frames, water_shell_layers)
        if isinstance(gcp1_output, ScalarVector):
            s_out1 = gcp1_output.scalar
            v_out1 = gcp1_output.vector
        else:
            s_out1 = gcp1_output
            v_out1 = None
        gcp2_output = self.gcp2((s_out1, v_out1), edge_index, frames, water_shell_layers)
        if isinstance(gcp2_output, ScalarVector):
            s_out2 = gcp2_output.scalar
            v_out2 = gcp2_output.vector
        else:
            s_out2 = gcp2_output
            v_out2 = None
        batched_graph.ndata['h'] = s_out2
        h = self.gat(batched_graph, s_out2)
        h = h.mean(dim=1)
        h_res = self.res_conv(s_out1)
        h = h + self.res_linear(h_res[:h.size(0)])
        if self.pooling_type == 'hierarchical':
            mean_feat = self.pool1(batched_graph, h)
            max_feat = self.pool2(batched_graph, h)
            sorted_feat = self.pool3(batched_graph, h)
            sorted_feat = sorted_feat.view(sorted_feat.size(0), -1)
            batched_feat = torch.cat([mean_feat, max_feat, sorted_feat], dim=1)
        else:
            batched_feat = self.pool(batched_graph, h)
        return self.mlp(batched_feat).squeeze(-1)


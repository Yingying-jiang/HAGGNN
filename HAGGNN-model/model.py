import torch
import torch.nn as nn
from hacdconvmodel1 import HACDConvModel
from haegnn1 import HAEGNN,EGNN
from sklearn.preprocessing import normalize
import numpy as np
import dgl

def orientation(pos):
    u = normalize(X=pos[1:,:] - pos[:-1,:], norm='l2', axis=1)
    u1 = u[1:,:]
    u2 = u[:-1, :]
    b = normalize(X=u2 - u1, norm='l2', axis=1)
    n = normalize(X=np.cross(u2, u1), norm='l2', axis=1)
    o = normalize(X=np.cross(b, n), norm='l2', axis=1)
    ori = np.stack([b, n, o], axis=1)
    ori_tensor = torch.tensor(np.concatenate([np.expand_dims(ori[0], 0), ori, np.expand_dims(ori[-1], 0)], axis=0),
                              dtype=torch.float32)
    return ori_tensor

class EnsembleModel(nn.Module):
    def __init__(self,
                 hacdconv_params,
                 egnn_params,
                 mlp_hidden_dims,
                 output_dim,
                 act_fn=nn.SiLU(),
                 pool_method="mean",
                 egnn_pool_hidden_dim=128,
                 egnn_pool_dropout=0.1):
        super().__init__()
        self.pool_method = pool_method
        self.egnn_pool_hidden_dim = egnn_pool_hidden_dim
        self.model2 = HACDConvModel(**hacdconv_params)
        self.model1 = EGNN(**egnn_params)

        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False

        self.egnn_pool = nn.Sequential(
            nn.Linear(egnn_params['out_node_nf'] * (2 if pool_method == "mean+max" else 1), 
                     egnn_pool_hidden_dim),
            nn.LayerNorm(egnn_pool_hidden_dim),
            act_fn,
            nn.Dropout(egnn_pool_dropout)
        )

        self.model2_out_dim = hacdconv_params['channels'][-1]
        self.model1_out_dim = egnn_pool_hidden_dim
        self.weight_logits = nn.Parameter(torch.zeros(2))
        self.final_mlp = nn.Sequential()
        in_dim = self.model1_out_dim + self.model2_out_dim
        
        for i, hidden_dim in enumerate(mlp_hidden_dims):
            self.final_mlp.add_module(
                f"layer_{i}",
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    act_fn,
                    nn.Dropout(0.1)
                )
            )
            in_dim = hidden_dim

        self.final_mlp.add_module(
            "output_layer",
            nn.Linear(in_dim, output_dim)
        )

        self.apply(self._init_weights)

    def forward(self, batched_graph, node_s, coords, seq, ori, edge_index, edge_attr, water_shell_layers):
        num_nodes_per_graph = [g.number_of_nodes() for g in dgl.unbatch(batched_graph)]
        batch_tensor = torch.cat([
        torch.full((n,), i, dtype=torch.long, device=node_s.device)
        for i, n in enumerate(num_nodes_per_graph)
        ])
        data_for_hacdconv = type('PseudoData',(),{
            'x': node_s,
            'pos': coords,
            'seq': seq,
            'ori': ori,
            'batch': batch_tensor,
            'water_shell_layers': water_shell_layers
        })
        out2 = self.model2(data_for_hacdconv)
        node_features, _ = self.model1(
            batched_graph,
            node_s,
            coords,
            edge_index,
            edge_attr,
            water_shell_layers
        )
        node_features_split = torch.split(node_features, batched_graph.batch_num_nodes().tolist())
        pooled_features = []
        
        for nf in node_features_split:
            if self.pool_method == "mean":
                pooled = torch.mean(nf, dim=0)
            elif self.pool_method == "max":
                pooled = torch.max(nf, dim=0)[0]
            elif self.pool_method == "mean+max":
                pooled = torch.cat([torch.mean(nf, dim=0), torch.max(nf, dim=0)[0]])
            pooled_features.append(pooled)
        
        pooled = torch.stack(pooled_features)
        out1 = self.egnn_pool(pooled)
        temperature = 0.3
        weights = torch.softmax(self.weight_logits / temperature, dim=0)
        weighted_out1 = weights[0] * out1
        weighted_out2 = weights[1] * out2
        combined = torch.cat([weighted_out1, weighted_out2], dim=1)
        return self.final_mlp(combined)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
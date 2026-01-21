import torch.nn as nn
from dgl import load_graphs
import dgl
from layer import GVP, GVPConvLayer, LayerNorm
import torch
import torch_scatter
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention

class ImprovedAggregation(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.weight_layer = nn.Linear(in_dim, 2)

    def forward(self, mean_feat, max_feat):
        combined = mean_feat + max_feat
        weights = torch.softmax(self.weight_layer(combined), dim=-1)
        return weights[:,0] * mean_feat + weights[:,1] * max_feat

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        return self.fc(x) + self.proj(x)

class MLPWithResidual(nn.Module):
    def __init__(self, in_dim, hidden_dims, output_dim, drop_rate=0.2):
        super().__init__()
        layers = []
        dim = in_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(drop_rate),
                ResidualBlock(hidden_dim, hidden_dim, drop_rate)
            ])
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class MLP(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim, dropout):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.SiLU(),
                nn.BatchNorm1d(out_dim),
                nn.Dropout(dropout)
            )
            self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            
        def forward(self, x):
            return self.block(x) + self.shortcut(x)
    
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=0.3):
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.SiLU())
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        
        for i in range(1, len(hidden_dims)):
            layers.append(
                self.ResBlock(
                    hidden_dims[i-1],
                    hidden_dims[i],
                    dropout
                )
            )
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

    
class GVPModel(nn.Module):
    def __init__(self, node_in_dim, node_h_dim,
                 edge_in_dim, edge_h_dim, num_layers, n_feedforward,  drop_rate):
        super().__init__()

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, n_feedforward=n_feedforward, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))
        
        gate_nn = nn.Sequential(
            nn.Linear(node_h_dim[0], 16),
            nn.LeakyReLU(0.1),
            nn.Dropout(drop_rate),
            nn.Linear(16, 1)
        )
        self.pooling = GlobalAttention(gate_nn=gate_nn)
        self.mlp = MLPWithResidual(node_h_dim[0], hidden_dims=[64, 32], output_dim=1)


    def forward(self, h_V, edge_index, h_E, water_shell, batch=None):
        device = 'cuda:0'
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E, water_shell=water_shell)
        out = self.W_out(h_V)
        out = out.to(device)
        batch = batch.to(device)
        global_context = torch_scatter.scatter_mean(out, batch, dim=0)
        global_context = self.context_encoder(global_context)
        global_context = global_context[batch]
        out = out + 0.5 * global_context
        graph_representation = self.pooling(out, batch)
        mlp_output = self.mlp(graph_representation)

        return mlp_output.view(-1,1)

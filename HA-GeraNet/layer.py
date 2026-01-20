import dgl
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add
from collections.abc import Sequence
from torchdrug import layers, core
from torchdrug.core import Registry as R


class GeometricRelationalGraphConv(nn.Module):
    eps = 1e-6
    def __init__(self, input_dim, hidden_dim, output_dim, edge_input_dim=61, kernel_hidden_dim=32,
                 dropout=0.05, dropout_before_conv=0.2, activation="relu", aggregate_func="sum"):
        super(GeometricRelationalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.kernel_hidden_dim = kernel_hidden_dim
        self.aggregate_func = aggregate_func
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.kernel = layers.MLP(edge_input_dim, [kernel_hidden_dim, (hidden_dim + 1) * hidden_dim])
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.input_batch_norm = nn.BatchNorm1d(input_dim)
        self.message_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout_before_conv = nn.Dropout(dropout_before_conv)
        self.activation = torch.nn.ReLU()
        self.shell_log_weights = nn.Parameter(torch.randn(3))

    def message(self, graph, node_features):
        all_messages = []
        for edge_type in ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6']:
            edge_features = graph.edges[('protein', edge_type, 'protein')].data['w']
            node_in, node_out = graph.edges(etype=('protein', edge_type, 'protein'))
            message = self.linear1(node_features[node_in])
            message = self.message_batch_norm(message)
            message = self.dropout_before_conv(self.activation(message))
            kernel = self.kernel(edge_features).view(-1, self.hidden_dim + 1, self.hidden_dim)
            message = torch.einsum('ijk, ik->ij', kernel[:, 1:, :], message) + kernel[:, 0, :]
            all_messages.append(message)
        final_message = torch.cat(all_messages, dim=0)

        return final_message

    def aggregate(self, graph, message, water_shell_layers_i, water_shell_layers_j):
        edge_weight = torch.ones(message.size(0), device=message.device)
        edge_weight = edge_weight.view(-1, 1).expand_as(message)
        water_shell_layers_j = water_shell_layers_j.long()
        all_node_out = []
        for edge_type in ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6']:
            edge_key = ('protein', edge_type, 'protein')
            if edge_key not in graph.canonical_etypes:
                raise KeyError(f"Edge type {edge_key} not found in graph")
            node_in, node_out = graph.edges(etype=edge_key)
            all_node_out.append(node_out)
        node_out_combined = torch.cat(all_node_out, dim=0).long()
        if node_out_combined.max() >= water_shell_layers_j.size(0):
            raise ValueError(f"Index {node_out_combined.max()} out of range for water_shell_layers_j of size {water_shell_layers_j.size(0)}")
        
        
        water_shell_layers_j = water_shell_layers_j.to(node_out_combined.device)
        shell_j = water_shell_layers_j[node_out_combined]
        index = (shell_j - 1).long()
        if index.min() < 0 or index.max() >= self.shell_log_weights.size(0):
            print(f"Warning: Shell index out of range: min={index.min()}, max={index.max()}")
            index = torch.clamp(index, 0, self.shell_log_weights.size(0) - 1)
        
        water_weight = self.shell_log_weights[index].unsqueeze(-1)
        water_edge_weight = edge_weight * (1 + water_weight)
        dim_size = graph.num_nodes()
        update = scatter_add(message * water_edge_weight, node_out_combined, dim=0, dim_size=dim_size)
        return update
    

    def combine(self, input, update):
        output = self.linear2(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    def forward(self, graph, node_features, water_shell_layers):
        water_shell_layers = water_shell_layers.long()
        message = self.message(graph, node_features)
        update = self.aggregate(graph, message, water_shell_layers, water_shell_layers)
        output = self.combine(input, update)
        return output


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=0.2):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class GearNetIEConv(nn.Module, core.Configurable):
    def __init__(self, input_dim, embedding_dim, hidden_dims, num_relation=7, edge_input_dim=61,
                 batch_norm=False, activation="relu", concat_hidden=False, short_cut=True,
                 readout="sum", dropout=0.2, layer_norm=False):
        super(GearNetIEConv, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [embedding_dim if embedding_dim > 0 else input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.concat_hidden = concat_hidden
        self.short_cut = short_cut
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.layer_norm = layer_norm
        if embedding_dim > 0:
            self.linear = nn.Linear(input_dim, embedding_dim)
            self.embedding_batch_norm = nn.BatchNorm1d(embedding_dim)

        self.layers = nn.ModuleList()
        self.ieconvs = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(GeometricRelationalGraphConv(
                input_dim=self.dims[i],
                hidden_dim=self.dims[i + 1],
                output_dim=self.dims[i + 1],
                edge_input_dim=61,
                dropout=0.05,
                kernel_hidden_dim=32,
                dropout_before_conv=0.2,
                activation="relu",
                aggregate_func="sum"
            ))
        if layer_norm:
            self.layer_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.layer_norms.append(nn.LayerNorm(self.dims[i + 1]))

        self.dropout = nn.Dropout(dropout)

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)
    def get_ieconv_edge_feature(self, graph):
        edge_feature = graph.edata['w']
        return edge_feature

    def forward(self, graph, node_features, water_shell_layers):
        hiddens = []
        layer_input = node_features

        if self.embedding_dim > 0:
            layer_input = self.linear(layer_input)
            layer_input = self.embedding_batch_norm(layer_input)
        edge_hidden = None
        ieconv_edge_feature = self.get_ieconv_edge_feature(graph)
        for i in range(len(self.layers)):
            hidden = self.layers[i](graph, layer_input, water_shell_layers)
            hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.layer_norm:
                hidden = self.layer_norms[i](hidden)
            hiddens.append(hidden)
            layer_input = hidden
        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        if node_feature.dim() == 2:
           node_feature = node_feature.unsqueeze(0)
        aggregated_features = node_features.sum(dim=0)
        return {
            "graph_feature": aggregated_features,
        }

class GearNet_MLP_Model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dims, num_relation,
                 batch_norm, activation, concat_hidden, short_cut, readout, dropout):
        super(GearNet_MLP_Model, self).__init__()
        self.gearnet = GearNetIEConv(input_dim, embedding_dim, hidden_dims, num_relation=num_relation,
                                     batch_norm=batch_norm, activation=activation,
                                     concat_hidden=concat_hidden, short_cut=short_cut,
                                     readout=readout, dropout=dropout)
        self.mlp = MLP(input_dim, hidden_dims=[512, 256, 128], output_dim=1)

    def forward(self, graph, node_features, water_shell_layers):
        gearnet_output = self.gearnet(graph, node_features, water_shell_layers)
        graph_features = gearnet_output["graph_feature"]
        batch_size = graph.batch_size
        graph_features = graph_features.unsqueeze(0).repeat(batch_size, 1)
        mlp_output = self.mlp(graph_features)
        
        return mlp_output.view(-1, 1)
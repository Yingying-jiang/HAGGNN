from torch import nn
import torch


class E_GCL(nn.Module):


    def __init__(self, input_nf, output_nf, hidden_nf, water_shell_layers, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1
        
        self.water_shell_layers = water_shell_layers
        self.shell_log_weights = nn.Parameter(torch.zeros(3))

        self.edge_mlp = nn.Sequential(
    nn.LayerNorm(input_edge + edge_coords_nf + edges_in_d),
    nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
    nn.SiLU(),
    nn.Dropout(0.2),
    nn.Linear(hidden_nf, hidden_nf),
    nn.SiLU()
)
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_index, radial, edge_attr, water_shell_layers):
        if torch.isnan(source).any() or torch.isinf(source).any():
            return torch.zeros_like(source)

        assert not torch.isnan(source).any() and not torch.isinf(source).any(), "sourc error"
        assert not torch.isnan(target).any() and not torch.isinf(target).any(), "target error"
        assert not torch.isnan(radial).any() and not torch.isinf(radial).any(), "radial error"
        if edge_attr is not None:
            assert not torch.isnan(edge_attr).any() and not torch.isinf(edge_attr).any(), "edge_attr error"
        batch_sizes = {
        'source': source.size(0),
        'target': target.size(0),
        'radial': radial.size(0),
        'edge_attr': edge_attr.size(0)
    }
    
        if len(set(batch_sizes.values())) > 1:
        if radial.size(0) != source.size(0):
            radial = radial.expand(source.size(0), radial.size(1))
        if edge_attr.size(0) != source.size(0):
            if edge_attr.size(0) > source.size(0):
                edge_attr = edge_attr[:source.size(0)]
            else:
                edge_attr = torch.nn.functional.pad(
                edge_attr, 
                (0, 0, 0, source.size(0) - edge_attr.size(0))
            )
    

        if source.dim() == 3:
            source = source.squeeze(1)
    
        if target.dim() == 3:
            target = target.squeeze(1)
    
        if radial.dim() == 3:
            radial = radial.squeeze(1)
    
        if edge_attr is not None and edge_attr.dim() == 3:
            edge_attr = edge_attr.squeeze(1)

        assert source.dim() == 2, f"source{source.dim()}"
        assert target.dim() == 2, f"target{target.dim()}"
        assert radial.dim() == 2, f"radial{radial.dim()}"
        if edge_attr is not None:
            assert edge_attr.dim() == 2, f"edge_attr{edge_attr.dim()}"

        num_edges = source.size(0)
        assert target.size(0) == num_edges, f"target{num_edges}，but{target.size(0)}"
        assert radial.size(0) == num_edges, f"radial{num_edges}，but{radial.size(0)}"
        if edge_attr is not None:
           assert edge_attr.size(0) == num_edges, f"edge_attr{num_edges}，but{edge_attr.size(0)}"

        try:
            step1 = torch.cat([source, target], dim=1)
            step2 = torch.cat([step1, radial], dim=1)
            out = torch.cat([step2, edge_attr], dim=1)
        except Exception as e:
            print(f"error: {e}")
            raise e
        row, col = edge_index
        if water_shell_layers.dim() == 1:
            water_shell_layers = water_shell_layers.unsqueeze(1)

        water_shell_i = water_shell_layers[row]
        water_shell_j = water_shell_layers[col]
        shell_i = water_shell_i.clamp(min=1).float()
        shell_j = water_shell_j.clamp(min=1).float()
        shell_layer = torch.maximum(shell_i, shell_j)
        index = (shell_layer - 1).long()
        weight_shell = self.shell_log_weights[index]
        out = out * (1 + weight_shell)
        out = self.edge_mlp(out)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)

        M = coord.size(0)
        C = 1.0 / (M - 1) if M > 1 else 1.0

        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + C * agg
        return coord


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[col]-coord[row]
        radial = torch.sum(coord_diff.pow(2), dim=1, keepdim=True)
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, water_shell_layers=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], edge_index, radial, edge_attr, water_shell_layers)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, water_shell_layers, out_node_nf, in_edge_nf=0, device='cuda:0', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.water_shell_layers = water_shell_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, self.water_shell_layers,edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr, water_shell_layers):
        h = self.embedding_in(h)
        x_init = x
    
        for i in range(0, self.n_layers):
            h_in = h
            x_in = x

            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, water_shell_layers=water_shell_layers)

            if i > 0:
                h = h_in + 0.5 * h

                if i < self.n_layers - 1:
                    x = x_in + 0.3 * (x - x_in)
    
        h = self.embedding_out(h)
        return h, x


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    if data.dim() == 1:
        data = data.unsqueeze(1)
    
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    
    result = data.new_full(result_shape, 0.0)
    count = data.new_full(result_shape, 0.0)
    
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))

    mean = result / count.clamp(min=1.0)
    return mean.squeeze(1) if data.dim() == 1 else mean

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = act_fn
        self.dropout = nn.Dropout(0.2)

        if in_dim != out_dim:
            self.residual_transform = nn.Linear(in_dim, out_dim)
        else:
            self.residual_transform = nn.Identity()

    def forward(self, x):
        residual = self.residual_transform(x)
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        out = 0.5*(x+residual)
        return out


class EGNNWithMLP(nn.Module):
    def __init__(self, egnn_params, mlp_hidden_dims, output_dim, act_fn=nn.SiLU()):
        super().__init__()

        self.egnn = EGNN(**egnn_params)
        egnn_out_dim = egnn_params['out_node_nf']
        mlp_input_dim = egnn_out_dim * 2
        self.input_layer = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dims[0]),
            nn.LayerNorm(mlp_hidden_dims[0]),
            act_fn,
            nn.Dropout(0.2)
        )
        self.mlp_layers = nn.ModuleList()
        for i in range(len(mlp_hidden_dims) - 1):
            in_dim = mlp_hidden_dims[i]
            out_dim = mlp_hidden_dims[i + 1]
            self.mlp_layers.append(ResidualBlock(in_dim, out_dim, act_fn))
        self.output_layer = nn.Linear(mlp_hidden_dims[-1], output_dim)
        self.apply(self._init_weights)

    def forward(self, graph, h, x, edge_index, edge_attr, water_shell_layers):
        h_egnn, x_egnn = self.egnn(h, x, edge_index, edge_attr, water_shell_layers)
        batch_num_nodes = graph.batch_num_nodes().tolist()
        h_egnn_list = torch.split(h_egnn, batch_num_nodes)
        h_pooled_list = [
            torch.cat([
                torch.mean(hg, dim=0, keepdim=True),
                torch.max(hg, dim=0, keepdim=True)[0]
            ], dim=-1)
            for hg in h_egnn_list
        ]
        h_pooled = torch.cat(h_pooled_list, dim=0)

        x = self.input_layer(h_pooled)
        for layer in self.mlp_layers:
            x = layer(x)
        output = self.output_layer(x)

        return output.squeeze(1)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


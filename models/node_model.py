from enum import Enum, auto

from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, FiLMConv, global_mean_pool, GATConv
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GATConv

from torch_geometric.utils import remove_self_loops
from torch.nn import functional as F

import torch
import torch.nn as nn

from .cooperative_sheaves.NSD.nsd_disc_models import DiscreteFlatBundleSheafDiffusion


class GINConv(nn.Module):
    def __init__(self, in_features, out_features, eps=0., train_eps=False):
        super(GINConv, self).__init__()
        self.eps = nn.Parameter(torch.tensor(eps), requires_grad=train_eps)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node feature matrix of shape (N, in_features)
            edge_index: Sparse adjacency matrix in COO format, shape (2, E)
            edge_weight: Edge weights of shape (E,). If None, treat as unweighted.
        """
        row, col = edge_index  # Extract source and target nodes

        # Handle edge weights (default to 1 if unweighted)
        if edge_weight is None:
            edge_weight = torch.ones_like(row, dtype=x.dtype, device=x.device)

        # Aggregate neighbor messages using edge weights
        agg_neighbors = torch.zeros_like(x)
        agg_neighbors = agg_neighbors.index_add(0, row, edge_weight.unsqueeze(-1) * x[col])

        # Update node features
        x_updated = (1 + self.eps) * x + agg_neighbors

        # Apply the MLP
        return self.mlp(x_updated)

class RGINConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features))))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.num_relations = args.num_relations
        self.layer_type = args.layer_type
        num_features = [args.input_dim] + list(args.hidden_layers) + [args.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()
    def get_layer(self, in_features, out_features):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(in_features, out_features)
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "GAT":
            return GATConv(in_features, out_features)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        for i, layer in enumerate(self.layers):
            if self.layer_type in ["R-GCN", "R-GIN"]:
                x = layer(x, edge_index, edge_type=graph.edge_type)
            else:
                x = layer(x, edge_index)
            if i != self.num_layers - 1:
                x = self.act_fn(x)
                x = self.dropout(x)
        return x


class GNN_TYPE(Enum):
    GCN = auto()
    GGNN = auto()
    GIN = auto()
    GAT = auto()
    CSN = auto()
    ONSD = auto()
    FNSD = auto()
    FGNSD = auto()

    @staticmethod
    def from_string(s):
        try:
            return GNN_TYPE[s]
        except KeyError:
            raise ValueError()

    def get_layer(self, args, in_dim, out_dim):
        if self is GNN_TYPE.GCN:
            return GCNConv(
                in_channels=in_dim,
                out_channels=out_dim)
        elif self is GNN_TYPE.GGNN:
            return GatedGraphConv(out_channels=out_dim, num_layers=1)
        elif self is GNN_TYPE.GIN:
            return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
                                         nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()))
        elif self is GNN_TYPE.GAT:
            # 4-heads, although the paper by Velickovic et al. had used 6-8 heads.
            # The output will be the concatenation of the heads, yielding a vector of size out_dim
            num_heads = 4
            return GATConv(in_dim, out_dim // num_heads, heads=num_heads)
        elif self is GNN_TYPE.FNSD:
            args.input_dim = in_dim
            args.output_dim = out_dim
            return DiscreteFlatBundleSheafDiffusion(args)

class FNSD(torch.nn.Module):
    def __init__(self, args, gnn_type, num_layers, dim0, h_dim, out_dim, last_layer_fully_adjacent,
                 unroll, layer_norm, use_activation, use_residual):
        super(FNSD, self).__init__()
        self.gnn_type = getattr(GNN_TYPE, gnn_type) if type(gnn_type) is str else gnn_type
        self.unroll = unroll
        self.last_layer_fully_adjacent = last_layer_fully_adjacent
        self.use_layer_norm = layer_norm
        self.use_activation = use_activation
        self.use_residual = use_residual
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.device = self.device

        self.num_layers = num_layers
        self.layer0_keys = nn.Embedding(num_embeddings=dim0 + 1, embedding_dim=h_dim*args.d)
        self.layer0_values = nn.Embedding(num_embeddings=dim0 + 1, embedding_dim=h_dim*args.d)
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        #self.lin1 = nn.Linear(h_dim, 32 * 2)
        #self.lin2 = nn.Linear(32 * 2, h_dim)

        if unroll:
            self.layers.append(self.gnn_type.get_layer(args,
                in_dim=h_dim,
                out_dim=h_dim))
        else:
            for i in range(num_layers):
                self.layers.append(self.gnn_type.get_layer(args,
                    in_dim=h_dim,
                    out_dim=h_dim))
        if self.use_layer_norm:
            for i in range(num_layers):
                self.layer_norms.append(nn.LayerNorm(h_dim*args.d))

        self.out_dim = out_dim
        # self.out_layer = nn.Linear(in_features=h_dim, out_features=out_dim, bias=False)
        self.out_layer = nn.Linear(in_features=h_dim*args.d, out_features=out_dim + 1, bias=False)

    def forward(self, data, reff=False):
        x, edge_index, batch, roots = data.x, data.edge_index, data.batch, data.root_mask

        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self.layer0_keys(x_key)
        x_val_embed = self.layer0_values(x_val)
        x = x_key_embed + x_val_embed

        get_layer_reff = [False] * self.num_layers
        if reff:
            get_layer_reff[-1] = True


        reff_per_layer = torch.zeros((self.num_layers,), device=self.device)
        for i in range(self.num_layers):
            if self.unroll:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            new_x = x
            if self.last_layer_fully_adjacent and i == self.num_layers - 1:
                root_indices = torch.nonzero(roots, as_tuple=False).squeeze(-1)
                target_roots = root_indices.index_select(dim=0, index=batch)
                source_nodes = torch.arange(0, data.num_nodes).to(self.device)
                edges = torch.stack([source_nodes, target_roots], dim=0)

            else:
                edges = edge_index

            #full_edges = torch.cat([edges, edges.flip(0)], dim=1).unique(dim=1)
            #edges = full_edges
            #edges = remove_self_loops(edges)[0]

            if 'Discrete' in layer.__class__.__name__ and 'Flat' not in layer.__class__.__name__:
                layer.get_edge_dependend_stuff(edges, new_x)
            # else:
            #     layer.compute_maps_idx(edges)
            #layer.to(new_x.device)

            if 'Flat' in layer.__class__.__name__:
                edges = remove_self_loops(edges)[0]
                new_x, reff_values = layer(new_x, edges, data, reff=get_layer_reff[i])
                reff_sum, mean_reff, std_reff = reff_values
                reff_per_layer[i] = reff_sum
            elif 'RGCN' in layer.__class__.__name__:
                new_x = layer(new_x, edges, edge_type=data.edge_type)
            else:
                new_x = layer(new_x, edges)
                
            if self.use_activation:
                new_x = F.relu(new_x)
            if self.use_residual:
                x = x + new_x
            else:
                x = new_x
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

        root_nodes = x[roots]
        logits = self.out_layer(root_nodes)
        # logits = F.linear(root_nodes, self.layer0_values.weight)
        if reff:
            return logits, reff_per_layer
        return logits

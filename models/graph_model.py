import torch
import torch.nn as nn
from measure_smoothing import dirichlet_normalized
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GATConv, PMLP, FiLMConv, global_mean_pool

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

class RGATConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GATConv(in_features, out_features))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

class RGINConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GINConv(in_features, out_features))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new




class GNN(torch.nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        self.num_relations = args.num_relations
        self.layer_type = args.layer_type
        num_features = [args.input_dim] + list(args.hidden_layers) + [args.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)
        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()

        if self.args.last_layer_fa:
            if self.layer_type == "R-GCN" or self.layer_type == "GCN":
                self.last_layer_transform = torch.nn.Linear(self.args.hidden_dim, self.args.output_dim)
            elif self.layer_type == "R-GIN" or self.layer_type == "GIN":
                self.last_layer_transform = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim),nn.BatchNorm1d(self.args.hidden_dim), nn.ReLU(),nn.Linear(self.args.hidden_dim, self.args.output_dim))
            else:
                raise NotImplementedError

    def get_layer(self, in_features, out_features):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GAT":
            return RGATConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(in_features, out_features)
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)

    def forward(self, graph, measure_dirichlet=False):
        x, edge_index, ptr, batch, edge_weight = graph.x, graph.edge_index, graph.ptr, graph.batch, graph.edge_weight
        x = x.float()

        for i, layer in enumerate(self.layers):
            if self.layer_type in ["R-GCN", "R-GAT", "R-GIN", "FiLM"]:
                x_new = layer(x, edge_index, edge_type=graph.edge_type)
            else:
                x_new = layer(x, edge_index, edge_weight=edge_weight)
            if i != self.num_layers - 1:
                x_new = self.act_fn(x_new)
                x_new = self.dropout(x_new)
            if i == self.num_layers - 1 and self.args.last_layer_fa:
                combined_values = global_mean_pool(x, batch)
                combined_values = self.last_layer_transform(combined_values)
                if self.layer_type in ["R-GCN", "R-GIN"]:
                    x_new += combined_values[batch]
                else:
                    x_new = combined_values[batch]
            x = x_new
        if measure_dirichlet:
            energy = dirichlet_normalized(x.cpu().numpy(), graph.edge_index.cpu().numpy())
            return energy
        x = global_mean_pool(x, batch)
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
        self.layer0 = nn.Linear(dim0, h_dim*args.d)
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
        self.out_layer = nn.Linear(in_features=h_dim*args.d, out_features=5, bias=False)

    
    def check_nan(self, tensor, name="Tensor"):
        if torch.isnan(tensor).any():
            print(f"{name} contains NaN values")
            exit(0)
        if torch.isinf(tensor).any():
            print(f"{name} contains Inf values")
            exit(0)

    def signal_propagation_metric(h_final, source_node, distances, stalk_dim):
        """
        Compute the signal propagation metric h_⊙^(m).
        
        Args:
            h_final: Final node features after m layers [num_nodes, stalk_dim]
            source_node: Index of the source node
            distances: Dictionary or matrix of graph distances from source node
            stalk_dim: Dimension of the stalk (feature dimension)
        
        Returns:
            h_metric: The signal propagation metric value
        """
        num_nodes = h_final.shape[0]
        
        # Find maximum distance from source for normalization
        max_distance = max(distances.values())
        
        total_propagation = 0.0
        
        # Sum over all feature dimensions and all nodes (except source)
        for f in range(stalk_dim):
            for u in range(num_nodes):
                if u == source_node:
                    continue
                    
                # Get the distance from source to node u
                d = distances.get(u, max_distance)  # Use max_distance if not found
                
                # Normalized feature contribution
                h_u_norm = torch.norm(h_final[u])
                if h_u_norm > 1e-8:  # Avoid division by zero
                    feature_contribution = abs(h_final[u, f]) / h_u_norm
                else:
                    feature_contribution = 0.0
                    
                total_propagation += feature_contribution * d
        
        # Normalize by feature dimension and max distance
        h_metric = total_propagation / (stalk_dim * max_distance)
        
        return h_metric.item()

    def compute_graph_distances(edge_index, num_nodes, source_node):
        """Compute shortest path distances from source node to all other nodes."""
        import networkx as nx
        # Convert to networkx graph for easy distance computation
        G = nx.Graph()
        G.add_edges_from(edge_index.t().tolist())
        
        distances = {}
        for node in range(num_nodes):
            try:
                distances[node] = nx.shortest_path_length(G, source=source_node, target=node)
            except:
                distances[node] = num_nodes  # Use large number if no path exists
        
        return distances

    def forward(self, data, reff=False):
        x, edge_index, data.edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # x_key_embed = self.layer0_keys(x_key)
        # x_val_embed = self.layer0_values(x_val)
        # x = x_key_embed + x_val_embed

        x = self.layer0(x)

        reff_per_layer = torch.zeros((self.num_layers,), device=self.device)
        for i in range(self.num_layers):
            if self.unroll:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            new_x = x
            # if self.last_layer_fully_adjacent and i == self.num_layers - 1:
            #     root_indices = torch.nonzero(roots, as_tuple=False).squeeze(-1)
            #     target_roots = root_indices.index_select(dim=0, index=batch)
            #     source_nodes = torch.arange(0, data.num_nodes).to(self.device)
            #     edges = torch.stack([source_nodes, target_roots], dim=0)

            # else:
            edges = edge_index

            full_edges = torch.cat([edges, edges.flip(0)], dim=1).unique(dim=1)
            edges = full_edges
            edges = remove_self_loops(edges)[0]

            if 'Discrete' in layer.__class__.__name__ and 'Flat' not in layer.__class__.__name__:
                layer.get_edge_dependend_stuff(edges, new_x)
            # else:
            #     layer.compute_maps_idx(edges)
            #layer.to(new_x.device)

            new_x, reff_values = layer(new_x, edges, data, reff=reff)

            reff_sum, mean_reff, std_reff = reff_values
            reff_per_layer[i] = reff_sum
                
            if self.use_activation:
                new_x = F.relu(new_x)
            if self.use_residual:
                x = x + new_x
            else:
                x = new_x
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

        
        # distances = self.compute_graph_distances(edge_index, data.num_nodes, source_node=0)

        # root_nodes = x[roots]
        logits = global_mean_pool(x, batch)
        # logits = self.out_layer(x)
        # logits = F.linear(root_nodes, self.layer0_values.weight)
        if reff:
            return logits, reff_per_layer
        return logits

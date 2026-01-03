import torch
import torch_sparse
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.models import GraphSAGE
from torch_scatter import scatter_add
from typing import Tuple
from .sheaf_base import SheafDiffusion
from .orthogonal import Orthogonal
from .sheaf_utils import get_layer_type, gnn_builder, custom_forward

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing

from torch_geometric.utils import degree

from .NSD import laplace as lap

class CoopSheafDiffusion(SheafDiffusion):
    def __init__(self, args):
        super(CoopSheafDiffusion, self).__init__(args)
        assert args['d'] > 1
        assert args['num_heads'] > 0

        #self.lin1 = nn.Linear(self.input_dim, self.hidden_channels * self.d)
        #self.pe_lin = nn.Linear(self.pe_size, self.pe_size*self.d)
        #self.lin2 = nn.Linear(self.hidden_channels * self.d, self.output_dim)

        # mlp_layers = []

        # if self.num_heads <= 1:
        #     mlp_layers.append(nn.Linear(self.hidden_channels * self.d, self.output_dim))
        # else:
        #     for i in range(self.num_heads-1):
        #         mlp_layers.append(nn.Linear(self.hidden_channels * self.d, self.hidden_channels * self.d))
        #         mlp_layers.append(nn.GELU())
        #     mlp_layers.append(nn.Linear(self.hidden_channels * self.d, self.output_dim))
        
        # self.mlp = nn.Sequential(*mlp_layers)

        self.coop_layers = nn.ModuleList(
            [NSDConv(args) for _ in range(self.layers)]
        )

        if self.layer_norm:
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm([self.hidden_channels * self.d]) for _ in range(self.layers)]
            )

        if self.batch_norm:
            self.batch_norm = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_channels * self.d, affine=True) for _ in range(self.layers)]
                )
        
    def compute_maps_idx(self, edge_index):
        left_right_idx, vertex_tril_idx, _ = compute_left_right_map_index_general(edge_index)
        self.left_idx, self.right_idx = left_right_idx
        self.tril_row, self.tril_col = vertex_tril_idx
        self.new_edge_index = torch.cat([vertex_tril_idx, vertex_tril_idx.flip(0)], dim=1)
        
        full_left_right_idx, _, _ = compute_left_right_map_index_general(edge_index, full_matrix=True)
        _, self.full_right_index = full_left_right_idx
        
        self.deg = degree(edge_index[0])

    def forward(self, x, edge_index):
        #x = x.float()
        #edge_index = edge_index.float() if edge_index is not None else None
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        #x = self.lin1(x)
        #pe = self.pe_lin(batch.pe) if batch.pe is not None else torch.empty(batch.x.size(0), 0, device=batch.x.device)
        #batch.x = torch.cat([batch.x, pe], dim=-1)

        #self.deg = degree(batch.edge_index[0], num_nodes=self.graph_size)

        if self.use_act:
            x = F.gelu(x)

        for layer in range(self.layers):
            #start = time.perf_counter()
            #x = self.coop_layers[layer](x, edge_index)
            x = self.coop_layers[layer](x, edge_index, self.left_idx,
                                              self.right_idx,
                                              self.tril_row,
                                              self.tril_col,
                                              self.deg,
                                              self.new_edge_index,
                                              self.full_right_index)
            #end = time.perf_counter()
            #print(f"Layer {layer} took {end - start:.4f} seconds")
            if self.layer_norm:
                x = self.layer_norms[layer](x)
            if self.batch_norm:
                x = self.batch_norm[layer](x)
        
        # if self.graph_level:
        #     x = global_mean_pool(x, batch)

        #x = self.mlp(x)

        return x
    
@torch.no_grad()
def compute_reverse_index(edge_index: torch.Tensor, num_nodes: int | None = None) -> torch.Tensor:
    """
    For each edge e=(u,v), returns the index of its reverse (v,u) if present, else -1.
    Works for directed/undirected, supports CUDA, O(E log E) without Python loops.

    Args:
        edge_index: LongTensor of shape [2, E].
        num_nodes:  Optional number of nodes. If None, inferred as max(edge_index)+1.

    Returns:
        rev_idx: LongTensor [E], rev_idx[e] = j or -1 if missing.
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    E = edge_index.size(1)
    device = edge_index.device
    if E == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0

    row, col = edge_index[0], edge_index[1]
    keys  = row * num_nodes + col          # key(u,v)
    rkeys = col * num_nodes + row          # key(v,u)

    sort_k, perm = torch.sort(keys)        # sort once
    pos = torch.searchsorted(sort_k, rkeys)
    pos = pos.clamp_max(sort_k.numel() - 1)

    cand = perm[pos]
    match = (sort_k.gather(0, pos) == rkeys)

    rev_idx = torch.full((E,), -1, dtype=torch.long, device=device)
    rev_idx[match] = cand[match]
    return rev_idx


@torch.no_grad()
def compute_left_right_map_index_general(
    edge_index: torch.Tensor,
    full_matrix: bool = False,
    *,
    num_nodes: int | None = None,
    strict: bool = False,
    include_self_loops: bool = False,
    drop_unpaired: bool = True,
):
    """
    Generalization of your function to ANY graph.

    Returns:
        left_right_index: LongTensor [2, K] with (left_edge_idx, right_edge_idx).
        new_edge_index:   LongTensor [2, K] containing the canonical edges.
        rev_idx:          LongTensor [E] reverse mapping for ALL edges (useful elsewhere).

    Behavior:
      - If full_matrix=False (default): we keep exactly one canonical orientation
        per *paired* edge. By default that is (u < v). Self-loops are included
        only if include_self_loops=True.
      - If full_matrix=True: we keep all edges; right_index is rev_idx. If the graph
        is directed and an edge has no reverse, right_index will be -1 unless
        drop_unpaired=True (default), in which case those are filtered out.
        Set strict=True to raise if any reverse is missing.
    """
    dev = edge_index.device
    E = edge_index.size(1)
    row, col = edge_index[0], edge_index[1]

    rev_idx = compute_reverse_index(edge_index, num_nodes=num_nodes)

    if full_matrix:
        left = torch.arange(E, device=dev)
        right = rev_idx

        if strict and (right < 0).any():
            missing = (right < 0).nonzero(as_tuple=False).view(-1)
            raise ValueError(f"{missing.numel()} edges have no reverse counterpart; set strict=False or fix input.")

        if drop_unpaired:
            keep = right >= 0
            left = left[keep]
            right = right[keep]

        new_edge_index = edge_index[:, left]

    else:
        # Keep ONE orientation per *paired* edge: canonical = row < col (and optional self-loops).
        mask = (row < col) & (rev_idx >= 0)
        if include_self_loops:
            mask = mask | ((row == col) & (rev_idx >= 0))

        left = mask.nonzero(as_tuple=False).view(-1)
        right = rev_idx[left]
        new_edge_index = edge_index[:, left]

    left_right_index = torch.stack([left, right], dim=0)
    return left_right_index, new_edge_index, rev_idx

class FlatSheafLearner(nn.Module):
    """Learns a flat sheaf passing node features through a GNN or MLP + activation."""

    def __init__(self, d: int,  hidden_channels: int, out_shape: Tuple[int], linear_emb: bool,
                 gnn_type: str, gnn_layers: int, gnn_hidden: int, gnn_default: bool, gnn_residual: bool,
                 pe_size: int, sheaf_act="tanh"):
        super(FlatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        assert (gnn_type, gnn_residual) != ('SGC', True), "SGC does not support residual connections."
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.gnn_layers = gnn_layers
        self.residual = gnn_residual
        self.gnn_default = gnn_default
        self.linear_emb = linear_emb
        self.gnn_hidden = gnn_hidden
        self.layer_type = gnn_type
        out_channels = int(np.prod(out_shape))

        if gnn_layers > 0:
            self.gnn = get_layer_type(gnn_type)
            # if gnn_default == 1:
            #     self.phi = GraphSAGE(
            #         hidden_channels * d + pe_size,
            #         gnn_hidden,
            #         num_layers=gnn_layers,
            #         out_channels=out_channels,
            #         norm='layer',
            #         )
            # elif gnn_default == 2:
            #     self.phi = GraphSAGE(
            #         hidden_channels * d + pe_size,
            #         gnn_hidden,
            #         num_layers=gnn_layers,
            #         out_channels=out_channels,
            #         dropout=0.2,
            #         act='gelu',
            #         project=True,
            #         bias=True,
            #         )
            # else:
            if linear_emb:
                self.emb1 = nn.Linear((hidden_channels + pe_size) * d, gnn_hidden)
                self.phi = gnn_builder(gnn_type, gnn_hidden, gnn_hidden, gnn_layers)
                self.emb2 = nn.Linear(gnn_hidden, out_channels)
            else:
                self.phi = gnn_builder(gnn_type, (hidden_channels + pe_size) * d, out_channels, gnn_layers, gnn_hidden)

        else:
            self.phi = torch.nn.Linear(hidden_channels, int(np.prod(out_shape)), bias=False)
            # self.phi = nn.Sequential(
            #     nn.Linear((hidden_channels + pe_size) * d, gnn_hidden),
            #     nn.ReLU(),
            #     nn.Linear(gnn_hidden, out_channels)
            # )

    def forward(self, x, edge_index):
        maps = custom_forward(self, x, edge_index)

        return torch.tanh(maps)

class NSDConv(SheafDiffusion, MessagePassing):
    def __init__(self,
                 args,
                 bias=False):
        SheafDiffusion.__init__(self, args)
        MessagePassing.__init__(self, aggr='add', flow='target_to_source', node_dim=0)

        if self.right_weights:
            self.lin_right_weights = nn.Linear(self.hidden_channels, self.hidden_channels, bias=bias)
            nn.init.orthogonal_(self.lin_right_weights.weight.data)
        else:
            self.lin_right_weights = nn.Identity()
        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.d, self.d, bias=bias)
            nn.init.eye_(self.lin_left_weights.weight.data)
        else:
            self.lin_left_weights = nn.Identity()

        self.orth_transform = Orthogonal(d=self.d, orthogonal_map=self.orth_trans)

        self.sheaf_learner = LocalConcatSheafLearnerVariant(
                self.d,
                self.hidden_channels,
                out_shape=(self.get_param_size(),),
                # linear_emb=self.linear_emb,
                # gnn_type=self.gnn_type,
                # gnn_layers=self.gnn_layers,
                # gnn_hidden=self.gnn_hidden,
                # gnn_default=self.gnn_default,
                # gnn_residual=self.gnn_residual,
                # pe_size=self.pe_size,
                # conformal=self.conformal,
                sheaf_act=self.sheaf_act)

        if self.use_edge_weights:
            self.weights_learner = EdgeWeightLearner(self.hidden_channels * self.d)

        self.epsilons = nn.Parameter(torch.zeros((self.d, 1)))
    
    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def restriction_maps_builder(self, F, edge_index, edge_weights, left_idx, right_idx, tril_row, tril_col, deg):#, edge_weights):
        row, _ = edge_index
        edge_weights = edge_weights.squeeze(-1) if edge_weights is not None else None

        maps = self.orth_transform(F)

        vertex_tril_idx = torch.cat([tril_row[None, :], tril_col[None, :]], dim=0)
        self.new_edge_index = torch.cat([vertex_tril_idx, vertex_tril_idx.flip(0)], dim=1)

        if edge_weights is not None:
            diag_maps = scatter_add(edge_weights ** 2, row, dim=0, dim_size=self.graph_size)
            maps = maps * edge_weights[:, None, None]
        else:
            diag_maps = degree(row, num_nodes=self.graph_size)

        left_maps = maps[left_idx]
        right_maps = maps[right_idx]

        diag_sqrt_inv = (diag_maps + 1).pow(-0.5)
        left_norm = diag_sqrt_inv[tril_row]
        right_norm = diag_sqrt_inv[tril_col]

        norm_left_maps = left_norm.view(-1, 1, 1) * left_maps
        norm_right_maps = right_norm.view(-1, 1, 1) * right_maps

        maps_prod = -torch.bmm(norm_left_maps.transpose(-2,-1), norm_right_maps)

        norm_D = diag_maps * diag_sqrt_inv**2 

        return norm_D, maps_prod

    # def restriction_maps_builder(self, F, edge_index, edge_weights, left_idx, right_idx, tril_row, tril_col, deg):#, edge_weights):
    #     row, _ = edge_index

    #     maps = self.orth_transform(F)

    #     if edge_weights is not None:
    #         diag_maps = scatter_add(edge_weights ** 2, row, dim=0, dim_size=self.graph_size)
    #         maps = maps * edge_weights[:, None, None]
    #     else:
    #         diag_maps = deg

    #     left_maps = maps[left_idx]
    #     right_maps = maps[right_idx]

    #     diag_sqrt_inv = (diag_maps + 1).pow(-0.5)
    #     left_norm = diag_sqrt_inv[tril_row]
    #     right_norm = diag_sqrt_inv[tril_col]

    #     norm_left_maps = left_norm.view(-1, 1, 1) * left_maps
    #     norm_right_maps = right_norm.view(-1, 1, 1) * right_maps

    #     maps_prod = -torch.bmm(norm_left_maps.transpose(-2,-1), norm_right_maps)

    #     norm_D = diag_maps * diag_sqrt_inv**2 

    #     return norm_D, maps_prod

    def left_right_linear(self, x, left, right):
        x = x.t().reshape(-1, self.d)
        x = left(x)
        x = x.reshape(-1, self.graph_size * self.d).t()

        x = right(x)
        return x

    def forward(self, x, edge_index, left_idx, right_idx, tril_row, tril_col, deg, new_edge_index, full_right_index):
        self.graph_size = x.size(0)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x_maps = x.reshape(self.graph_size, self.hidden_channels * self.d)

        to_be_F_maps = self.sheaf_learner(x_maps, edge_index)
        edge_weights = self.weights_learner(x_maps, edge_index, full_right_index) if self.use_edge_weights else None

        x = x.view(self.graph_size * self.d, -1)
        x0 = x

        D, prod = self.restriction_maps_builder(to_be_F_maps,
                                                edge_index,
                                                edge_weights,
                                                left_idx,
                                                right_idx,
                                                tril_row,
                                                tril_col,
                                                deg)

        x = self.left_right_linear(x, self.lin_left_weights, self.lin_right_weights)
        x = x.reshape(self.graph_size, self.d, self.hidden_channels)

        maps_prod = torch.cat([prod, prod.transpose(-2,-1)],dim=0)

        Dx = D[:, None, None] * x * deg.pow(-1)[:, None, None]
        #Dx = D[..., None] * batch.x

        self.propagate(new_edge_index, x=x, diag=Dx, FtF=maps_prod)
        #x = F.dropout(x, p=self.dropout, training=self.training)

        if self.use_act:
            x = F.gelu(x)   

        x = x.reshape(self.graph_size * self.d, -1)
        x0 = (1 + torch.tanh(self.epsilons).tile(self.graph_size, 1)) * x0 - x
        x = x0

        return x.view(self.graph_size, -1)

    def message(self, x_j, diag_i, FtF):
    #def message(self, x_i, x_j, D_i, F_row, F_col):
        #diag = D_i[:, None, None] * x_i
        msg = FtF @ x_j
        #msg = F_row.transpose(-2,-1) @ F_col @ x_j

        return diag_i + msg
    
class LocalConcatSheafLearnerVariant(nn.Module):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, d: int, hidden_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"):
        super(LocalConcatSheafLearnerVariant, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(hidden_channels * 2, int(np.prod(out_shape)), bias=False)
        # self.linear2 = torch.nn.Linear(self.d, 1, bias=False)

        # std1 = 1.414 * math.sqrt(2. / (hidden_channels * 2 + 1))
        # std2 = 1.414 * math.sqrt(2. / (d + 1))
        #
        # nn.init.normal_(self.linear1.weight, 0.0, std1)
        # nn.init.normal_(self.linear2.weight, 0.0, std2)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index):
        row, col = edge_index

        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        x_cat = torch.cat([x_row, x_col], dim=-1)
        x_cat = x_cat.reshape(-1, self.d, self.hidden_channels * 2).sum(dim=1)

        x_cat = self.linear1(x_cat)

        # x_cat = x_cat.t().reshape(-1, self.d)
        # x_cat = self.linear2(x_cat)
        # x_cat = x_cat.reshape(-1, edge_index.size(1)).t()

        maps = self.act(x_cat)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])
        
class EdgeWeightLearner(nn.Module):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int):
        super(EdgeWeightLearner, self).__init__()
        self.in_channels = in_channels
        self.linear1 = torch.nn.Linear(in_channels*2, 1, bias=False)
        #self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)

    def forward(self, x, edge_index, left_right_idx):

        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        weights = self.linear1(torch.cat([x_row, x_col], dim=1))
        weights = torch.sigmoid(weights)

        edge_weights = weights * torch.index_select(weights, index=left_right_idx, dim=0)
        return edge_weights.squeeze(-1)
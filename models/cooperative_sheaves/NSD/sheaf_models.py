# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from abc import abstractmethod
from torch import nn
from . import laplace as lap

from ..sheaf_utils import get_layer_type, gnn_builder, custom_forward


class SheafLearner(nn.Module):
    """Base model that learns a sheaf from the features and the graph structure."""
    def __init__(self):
        super(SheafLearner, self).__init__()
        self.L = None

    @abstractmethod
    def forward(self, x, edge_index):
        raise NotImplementedError()

    def set_L(self, weights):
        self.L = weights.clone().detach()


class LocalConcatSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"):
        super(LocalConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(in_channels*2, int(np.prod(out_shape)), bias=False)

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
        maps = self.linear1(torch.cat([x_row, x_col], dim=1))
        maps = self.act(maps)

        # sign = maps.sign()
        # maps = maps.abs().clamp(0.05, 1.0) * sign

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])


class LocalConcatSheafLearnerVariant(SheafLearner):
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
        
class LocalConcatFlatSheafLearnerVariant(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, d: int, hidden_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"):
        super(LocalConcatFlatSheafLearnerVariant, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(hidden_channels, int(np.prod(out_shape)), bias=False)
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
        # row, col = edge_index

        # x_row = torch.index_select(x, dim=0, index=row)
        # x_col = torch.index_select(x, dim=0, index=col)
        # x_cat = torch.cat([x_row, x_col], dim=-1)
        x_cat = x.reshape(-1, self.d, self.hidden_channels).sum(dim=1)

        x_cat = self.linear1(x_cat)

        # x_cat = x_cat.t().reshape(-1, self.d)
        # x_cat = self.linear2(x_cat)
        # x_cat = x_cat.reshape(-1, edge_index.size(1)).t()

        maps = self.act(x_cat)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

class AttentionSheafLearner(SheafLearner):

    def __init__(self, in_channels, d):
        super(AttentionSheafLearner, self).__init__()
        self.d = d
        self.linear1 = torch.nn.Linear(in_channels*2, d**2, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.linear1(torch.cat([x_row, x_col], dim=1)).view(-1, self.d, self.d)

        id = torch.eye(self.d, device=edge_index.device, dtype=maps.dtype).unsqueeze(0)
        return id - torch.softmax(maps, dim=-1)


class EdgeWeightLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, edge_index):
        super(EdgeWeightLearner, self).__init__()
        self.in_channels = in_channels
        self.linear1 = torch.nn.Linear(in_channels*2, 1, bias=False)
        #self.full_left_right_idx, _, _ = compute_left_right_map_index_general(edge_index, full_matrix=True)

    def forward(self, x, edge_index, full_left_right_idx):

        row, col = edge_index

        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        weights = self.linear1(torch.cat([x_row, x_col], dim=1))
        weights = torch.sigmoid(weights)

        edge_weights = weights * torch.index_select(weights, index=full_left_right_idx, dim=0)
        return edge_weights

    def update_edge_index(self, edge_index):
        self.full_left_right_idx, _, _ = compute_left_right_map_index_general(edge_index, full_matrix=True)


class QuadraticFormSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, out_shape: Tuple[int]):
        super(QuadraticFormSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape

        tensor = torch.eye(in_channels).unsqueeze(0).tile(int(np.prod(out_shape)), 1, 1)
        self.tensor = nn.Parameter(tensor)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.map_builder(torch.cat([x_row, x_col], dim=1))

        if len(self.out_shape) == 2:
            return torch.tanh(maps).view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return torch.tanh(maps).view(-1, self.out_shape[0])

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
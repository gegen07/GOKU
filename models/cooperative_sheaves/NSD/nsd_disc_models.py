# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch_sparse

from torch import nn
from .sheaf_base import SheafDiffusion
from . import laplacian_builders as lb
from models.cooperative_sheaves.NSD.sheaf_models import (
    LocalConcatSheafLearner,
    EdgeWeightLearner,
    LocalConcatSheafLearnerVariant,
    LocalConcatFlatSheafLearnerVariant,
    FlatSheafLearner,
)

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter_add

from . import laplace as lap
from ..orthogonal import Orthogonal

import numpy as np

class DiscreteDiagSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteDiagSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 0

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d,), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                         normalised=self.normalised,
                                                         deg_normalised=self.deg_normalised,
                                                         add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

class DiscreteBundleSheafDiffusion(SheafDiffusion, MessagePassing):

    def __init__(self, args):
        #super(DiscreteBundleSheafDiffusion, self).__init__(args)
        SheafDiffusion.__init__(self, args)
        MessagePassing.__init__(self, aggr='add', flow='target_to_source', node_dim=0)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        for i in range(self.layers):
            if self.right_weights:
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
            else:
                self.lin_right_weights.append(nn.Identity())
            if self.left_weights:
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)
            else:
                self.lin_left_weights.append(nn.Identity())

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        #self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        #self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.orth_transform = Orthogonal(self.d, self.orth_trans)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x
    
    def get_edge_dependend_stuff(self, edge_index, x):
        self.graph_size = x.size(0)
        undirected_edges = torch.cat([edge_index, edge_index.flip(0)], dim=1).unique(dim=1)

        if self.use_edge_weights:
            for _ in range(self.layers):
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, undirected_edges))
            self.weight_learners.to(self.device)

        # self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
        #     self.graph_size, undirected_edges, d=self.d, add_hp=self.add_hp,
        #     add_lp=self.add_lp, orth_map=self.orth_trans)
        
        self.left_right_idx, self.vertex_tril_idx = lap.compute_left_right_map_index(undirected_edges)
        self.left_idx, self.right_idx = self.left_right_idx
        self.tril_row, self.tril_col = self.vertex_tril_idx
        self.new_edge_index = torch.cat([self.vertex_tril_idx, self.vertex_tril_idx.flip(0)], dim=1)

        full_left_right_idx, _ = lap.compute_left_right_map_index(undirected_edges, full_matrix=True)
        _, self.full_right_index = full_left_right_idx
        
        self.deg = degree(edge_index[0])

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def align_edges_with_dummy(self, edge_index, num_edges=None):
        row, col = edge_index
        if num_edges is None:
            num_edges = edge_index.size(1) // 2
        N = num_edges
        dummy = self.graph_size 

        a = torch.minimum(row, col)
        b = torch.maximum(row, col)
        key = a * (N + 1) + b

        uniq, inv = torch.unique(key, return_inverse=True)
        K = uniq.numel()

        row_gt = torch.full((K,), dummy, device=row.device, dtype=torch.long)
        col_gt = torch.full((K,), dummy, device=row.device, dtype=torch.long)
        row_lt = torch.full((K,), dummy, device=row.device, dtype=torch.long)
        col_lt = torch.full((K,), dummy, device=row.device, dtype=torch.long)

        m_gt = row > col
        m_lt = row < col

        idx_gt = inv[m_gt]
        row_gt[idx_gt] = row[m_gt]
        col_gt[idx_gt] = col[m_gt]

        idx_lt = inv[m_lt]
        row_lt[idx_lt] = row[m_lt]
        col_lt[idx_lt] = col[m_lt]

        edge_index_gt_aligned = torch.stack([row_gt, col_gt], dim=0)
        edge_index_lt_aligned = torch.stack([row_lt, col_lt], dim=0)

        return edge_index_gt_aligned, edge_index_lt_aligned, dummy
    
    def restriction_maps_builder(self, maps, edge_index, edge_weights):
        row, _ = edge_index
        edge_weights = edge_weights.squeeze(-1) if edge_weights is not None else None
        maps = self.orth_transform(maps)

        if edge_weights is not None:
            diag_maps = scatter_add(edge_weights ** 2, row, dim=0, dim_size=self.graph_size)
            maps = maps * edge_weights[:, None, None]
        else:
            diag_maps = degree(row, num_nodes=self.graph_size)

        left_maps = maps[self.left_idx]
        right_maps = maps[self.right_idx]

        diag_sqrt_inv = (diag_maps + 1).pow(-0.5)
        left_norm = diag_sqrt_inv[self.tril_row]
        right_norm = diag_sqrt_inv[self.tril_col]

        norm_left_maps = left_norm.view(-1, 1, 1) * left_maps
        norm_right_maps = right_norm.view(-1, 1, 1) * right_maps

        maps_prod = -torch.bmm(norm_left_maps.transpose(-2,-1), norm_right_maps)
        
        norm_D = diag_maps * diag_sqrt_inv**2 

        return norm_D, maps_prod
    
    def align_edges_and_maps(self, edge_index, maps, dummy):
        dummies = torch.ones_like(edge_index) * dummy
        mask = edge_index != dummies

        edge_index = torch.cat([edge_index[0][mask[0]][None, :],
                                edge_index[1][mask[1]][None, :]], dim=0)

        maps = maps[mask[0]]

        return edge_index, maps

    def forward(self, x, edge_index, data, reff=False):
        torch.set_printoptions(linewidth=200)
        #x = x.to(torch.float64)
        self.edge_index = edge_index
        undirected_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).unique(dim=1)

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        #x = self.lin1(x)

        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        upper_edge_index, lower_edge_index, dummy = self.align_edges_with_dummy(edge_index)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](x_maps, undirected_edge_index)
                edge_weights = self.weight_learners[layer](x_maps, undirected_edge_index) if self.use_edge_weights else None
                #L, trans_maps = self.laplacian_builder(maps, edge_weights)
                #self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            D, maps_prod = self.restriction_maps_builder(maps, undirected_edge_index, edge_weights)

            y = x.clone()
            y = y.reshape(self.graph_size, self.d, self.hidden_channels)
            deg = degree(undirected_edge_index[0], num_nodes=self.graph_size)
            Dx = D[:, None, None] * y * deg.pow(-1)[:, None, None]

            if edge_index.size(1) != undirected_edge_index.size(1):
                u_edge_index, u_maps_prod = self.align_edges_and_maps(upper_edge_index, maps_prod, dummy)
                l_edge_index, l_maps_prod = self.align_edges_and_maps(lower_edge_index, maps_prod, dummy)

                y1 = self.propagate(u_edge_index, x=y, diag=Dx, Ft=u_maps_prod.transpose(-2,-1))
                y2 = self.propagate(l_edge_index, x=y, diag=Dx, Ft=l_maps_prod)
                y = y1 + y2
            else:
                y1 = self.propagate(upper_edge_index, x=y, diag=Dx, Ft=maps_prod.transpose(-2,-1))
                y2 = self.propagate(lower_edge_index, x=y, diag=Dx, Ft=maps_prod)
                y = y1 + y2

            y = y.view(self.graph_size * self.final_d, -1)

            # Use the adjacency matrix rather than the diagonal
            # x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            # print(torch.allclose(x,y, atol=1e-6))
            # print(torch.norm(x-y, p=2))

            x = y

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        sum_reff, mean_reff, var_reff = 0, 0, 0

        x = x.reshape(self.graph_size, -1)
        #x = self.lin2(x)
        return x, (sum_reff, mean_reff, var_reff)#F.log_softmax(x, dim=1)

    def message(self, x_j, diag_i, Ft):
        msg = Ft @ x_j

        return diag_i + msg

class DiscreteFlatBundleSheafDiffusion(SheafDiffusion, MessagePassing):

    def __init__(self, args):
        #super(DiscreteBundleSheafDiffusion, self).__init__(args)
        SheafDiffusion.__init__(self, args)
        MessagePassing.__init__(self, aggr='add', flow='source_to_target', node_dim=0)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            self.sheaf_learners.append(
                FlatSheafLearner(
                    self.d,
                    self.hidden_channels,
                    out_shape=(self.get_param_size(),),
                    linear_emb=self.linear_emb,
                    gnn_type=self.gnn_type,
                    gnn_layers=self.gnn_layers,
                    gnn_hidden=self.gnn_hidden,
                    gnn_default=self.gnn_default,
                    gnn_residual=self.gnn_residual,
                    pe_size=self.pe_size,
                    sheaf_act=self.sheaf_act)
            )

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        #self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.orth_transform = Orthogonal(self.d, self.orth_trans)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x
    
    def restriction_maps_builder(self, F, edge_index):
        row, _ = edge_index

        maps = self.orth_transform(F)

        diag_maps = degree(row, num_nodes=self.graph_size)

        diag_sqrt_inv = (diag_maps + 1).pow(-0.5)

        norm_maps = diag_sqrt_inv.view(-1, 1, 1) * maps

        norm_D = diag_maps * diag_sqrt_inv**2 

        return norm_D, norm_maps, maps

    def forward(self, x, edge_index, data, reff=False):
        self.graph_size = x.size(0)
        self.edge_index = edge_index
        self.undirected_edges = torch.cat([edge_index, edge_index.flip(0)], dim=1).unique(dim=1)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        #x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](x_maps, self.edge_index)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])
            
            D, maps, unnormalized_maps = self.restriction_maps_builder(maps, edge_index)

            y = x.clone()
            y = y.reshape(self.graph_size, self.d, self.hidden_channels)
            deg = degree(self.undirected_edges[0], num_nodes=self.graph_size)
            Dx = D[:, None, None] * y * deg.pow(-1)[:, None, None]
            Fy = maps @ y
            y = self.propagate(edge_index, x=Fy, diag=Dx, Ft=maps.transpose(-2,-1))
            
            x = y.view(self.graph_size * self.final_d, -1)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        sum_reff, mean_reff, var_reff = 0, 0, 0
        torch_R_F = torch.tensor([0.], dtype=torch.float64, device=x.device)

        x = x.reshape(self.graph_size, -1)
        #x = self.lin2(x)
        return x, (torch_R_F, mean_reff, var_reff)#F.log_softmax(x, dim=1)

    def message(self, x_j, diag_i, Ft_i):
        msg = Ft_i @ x_j

        return diag_i - msg

class DiscreteGeneralSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteGeneralSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
    
    def restriction_maps_builder(self, maps, edge_index):
        row, _ = edge_index
        maps = maps.view(-1, self.d, self.d)

        diag_maps = scatter_add(maps.transpose(-2,-1) @ maps,
                                row, dim=0, dim_size=self.graph_size)
        # print(f"These are the MP unnormalized diag maps: \n {diag_maps}")

        left_maps = maps[self.left_idx]
        right_maps = maps[self.right_idx]
        non_diag_maps = -torch.bmm(left_maps.transpose(-2,-1), right_maps)
        # print(f"These are the MP non diag maps unnormalized: \n {non_diag_maps}")

        if self.training:
            # During training, we perturb the matrices to ensure they have different singular values.
            # Without this, the gradients of batched_sym_matrix_pow, which uses SVD are non-finite.
            eps = torch.FloatTensor(self.d).uniform_(-0.001, 0.001).to(device=self.device)
        else:
            eps = torch.zeros(self.d, device=self.device)

        to_be_inv_diag_maps = diag_maps #+ torch.diag(1. + eps).unsqueeze(0) #if self.augmented else diag_maps
        diag_sqrt_inv = lap.batched_sym_matrix_pow(to_be_inv_diag_maps, -0.5)
        left_norm = diag_sqrt_inv[self.tril_row]
        right_norm = diag_sqrt_inv[self.tril_col]

        non_diag_maps = (left_norm @ non_diag_maps @ right_norm).clamp(min=-1, max=1)

        norm_D = (diag_sqrt_inv @ diag_maps @ diag_sqrt_inv).clamp(min=-1, max=1)

        # print(f"These are the MP normalized diag maps: \n {norm_D}")
        # print(f"These are the MP non diag maps unnormalized: \n {non_diag_maps}")

        return norm_D, non_diag_maps

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

class DiscreteFlatGeneralSheafDiffusion(SheafDiffusion, MessagePassing):

    def __init__(self, args):
        #super(DiscreteBundleSheafDiffusion, self).__init__(args)
        SheafDiffusion.__init__(self, args)
        MessagePassing.__init__(self, aggr='add', flow='source_to_target', node_dim=0)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                # self.sheaf_learners.append(LocalConcatFlatSheafLearnerVariant(self.final_d,
                #     self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
                self.sheaf_learners.append(
                    FlatSheafLearner(
                        self.d,
                        self.hidden_channels,
                        out_shape=(self.d**2,),
                        linear_emb=self.linear_emb,
                        gnn_type=self.gnn_type,
                        gnn_layers=self.gnn_layers,
                        gnn_hidden=self.gnn_hidden,
                        gnn_default=self.gnn_default,
                        gnn_residual=self.gnn_residual,
                        pe_size=self.pe_size,
                        conformal=False,
                        sheaf_act=self.sheaf_act)
                )
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        #self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x
    
    def restriction_maps_builder(self, maps, edge_index):
        row, _ = edge_index

        maps, _ = maps
        maps = maps.view(-1, self.d, self.d)

        deg = degree(row, num_nodes=self.graph_size) + 1

        diag_maps = (maps.transpose(-2,-1) @ maps) * deg.view(-1, 1, 1)

        if self.training:
            # During training, we perturb the matrices to ensure they have different singular values.
            # Without this, the gradients of batched_sym_matrix_pow, which uses SVD are non-finite.
            eps = torch.FloatTensor(self.d).uniform_(-0.001, 0.001).to(device=self.device)
        else:
            eps = torch.zeros(self.d, device=self.device)

        to_be_inv_diag_maps = diag_maps + torch.diag(1. + eps).unsqueeze(0) #if self.augmented else diag_maps
        diag_sqrt_inv = lap.batched_sym_matrix_pow(to_be_inv_diag_maps, -0.5)

        norm_D = (diag_sqrt_inv @ diag_maps @ diag_sqrt_inv).clamp(min=-1, max=1)

        return norm_D, maps, diag_sqrt_inv

    def forward(self, x, edge_index, data, reff=False):
        self.graph_size = x.size(0)
        self.edge_index = edge_index
        self.undirected_edges = torch.cat([edge_index, edge_index.flip(0)], dim=1).unique(dim=1)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        #x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](x_maps, self.edge_index)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            D, maps, diag_sqrt_inv = self.restriction_maps_builder(maps, edge_index)

            x = x.reshape(self.graph_size, self.d, self.hidden_channels)
            deg = degree(edge_index[0], num_nodes=self.graph_size) + 1
            Dx = D @ x * deg.pow(-1)[:, None, None]
            Fx = (maps @ diag_sqrt_inv).clamp(min=-1, max=1) @ x
            x = self.propagate(edge_index, x=Fx, diag=Dx, Ft=(diag_sqrt_inv @ maps.transpose(-2,-1)).clamp(min=-1, max=1))
            
            x = x.view(self.graph_size * self.final_d, -1)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        sum_reff, mean_reff, var_reff = 0, 0, 0
        torch_R_F = torch.tensor([0.], dtype=torch.float64, device=x.device)

        x = x.reshape(self.graph_size, -1)
        #x = self.lin2(x)
        return x, (torch_R_F, mean_reff, var_reff)#F.log_softmax(x, dim=1)

    def message(self, x_j, diag_i, Ft_i):
        msg = Ft_i @ x_j

        return diag_i - msg
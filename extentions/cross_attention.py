# -*- coding: utf-8 -*-
"""
Hierarchical cross-attention block (Stage 2 + Stage 3 of Hierarchical-SAT).

Stage 2: Macro-GCN propagates information among chunk-level embeddings.
Stage 3: Each node cross-attends to all chunks within its own graph,
         then a learnable gate fuses local node features with the
         retrieved global chunk-level context.

This module is inserted AFTER the SAT encoder and BEFORE graph-level pooling.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean, scatter


class HierarchicalCrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(d_model)

        # Stage 2 -- one macro-GCN layer with edge weights
        self.macro_gcn = GCNConv(
            d_model, d_model,
            add_self_loops=True, normalize=True
        )

        # Stage 3 -- single-head cross-attention projections (kept simple)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Gated fusion (mirrors the Extension-3 residual gate pattern)
        self.W_gate = nn.Linear(2 * d_model, d_model)

        # Final BN, consistent with the SAT encoder's BN choice
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x, chunk_id, chunk_edge_index, chunk_edge_weight,
                node_batch):
        # ---- Stage 2 : chunk init via mean over assigned nodes ----
        Z = scatter_mean(x, chunk_id, dim=0)                       # [C, d]
        chunk_batch = scatter(node_batch, chunk_id, dim=0,
                              reduce='max', dim_size=Z.size(0))    # [C]

        if chunk_edge_index.numel() > 0:
            Z = F.relu(self.macro_gcn(Z, chunk_edge_index, chunk_edge_weight))
        else:
            # No inter-chunk edges in the entire batch -- rely on self-loops
            Z = F.relu(self.macro_gcn(Z, chunk_edge_index))

        # ---- Stage 3 : node-to-chunk cross-attention ----
        Q = self.W_Q(x)                                            # [N, d]
        K = self.W_K(Z)                                            # [C, d]
        V = self.W_V(Z)                                            # [C, d]

        # Each node attends ONLY to chunks within its own graph
        same_graph = node_batch.unsqueeze(1) == chunk_batch.unsqueeze(0)
        scores = (Q @ K.t()) * self.scale                          # [N, C]
        scores = scores.masked_fill(~same_graph, float('-inf'))
        attn = F.softmax(scores, dim=-1)                           # [N, C]
        G = attn @ V                                               # [N, d]

        # ---- Gated fusion ----
        gate = torch.sigmoid(self.W_gate(torch.cat([x, G], dim=-1)))
        x_out = gate * G + (1.0 - gate) * x
        x_out = self.norm(x_out)
        return x_out

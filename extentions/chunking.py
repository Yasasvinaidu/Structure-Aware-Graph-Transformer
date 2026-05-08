# -*- coding: utf-8 -*-
"""
Hierarchical chunking utilities for Hierarchical-SAT.

Partitions a graph into balanced chunks via greedy BFS and builds a
chunk-level macro-graph whose edge weights count the number of original
edges crossing each chunk boundary.

We use a self-contained BFS partitioner instead of METIS to avoid an
external native dependency. For ZINC-scale graphs (N <= 50) this is
both fast and adequate.
"""
import math
import torch


def _adjacency_list(edge_index, num_nodes):
    adj = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].append(d)
    return adj


def balanced_bfs_partition(edge_index, num_nodes, num_parts):
    """Greedy BFS-based balanced graph partitioning."""
    if num_parts <= 1 or num_nodes <= num_parts:
        return torch.zeros(num_nodes, dtype=torch.long)

    target_size = math.ceil(num_nodes / num_parts)
    chunk_id = torch.full((num_nodes,), -1, dtype=torch.long)

    adj = _adjacency_list(edge_index, num_nodes)
    deg = torch.tensor([len(a) for a in adj], dtype=torch.long)

    for c in range(num_parts):
        unassigned = (chunk_id == -1).nonzero(as_tuple=False).squeeze(-1)
        if unassigned.numel() == 0:
            break
        seed = unassigned[deg[unassigned].argmax()].item()

        queue = [seed]
        chunk_id[seed] = c
        count = 1
        head = 0
        while head < len(queue) and count < target_size:
            u = queue[head]
            head += 1
            for v in adj[u]:
                if chunk_id[v] == -1:
                    chunk_id[v] = c
                    count += 1
                    queue.append(v)
                    if count >= target_size:
                        break

    chunk_id[chunk_id == -1] = num_parts - 1
    return chunk_id


def build_macro_graph(edge_index, chunk_id, num_chunks):
    """Edge weight (a, b) = number of original edges crossing between chunks a and b."""
    src, dst = edge_index[0], edge_index[1]
    src_c = chunk_id[src]
    dst_c = chunk_id[dst]
    mask = src_c != dst_c

    if not mask.any():
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float))

    cross_src = src_c[mask]
    cross_dst = dst_c[mask]
    pair_key = cross_src * num_chunks + cross_dst
    unique_keys, counts = torch.unique(pair_key, return_counts=True)
    macro_src = unique_keys // num_chunks
    macro_dst = unique_keys % num_chunks
    macro_edge_index = torch.stack([macro_src, macro_dst], dim=0).long()
    macro_edge_weight = counts.float()
    return macro_edge_index, macro_edge_weight


def compute_chunks(edge_index, num_nodes, chunk_size, max_chunks):
    """End-to-end chunk + macro-graph computation for one graph."""
    num_chunks = max(2, min(max_chunks, math.ceil(num_nodes / chunk_size)))
    num_chunks = min(num_chunks, num_nodes)

    chunk_id = balanced_bfs_partition(edge_index, num_nodes, num_chunks)
    macro_edge_index, macro_edge_weight = build_macro_graph(
        edge_index, chunk_id, num_chunks
    )
    return chunk_id, macro_edge_index, macro_edge_weight, num_chunks

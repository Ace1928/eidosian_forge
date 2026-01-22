from typing import Any, Dict, List, Mapping
import numpy as np
import torch
from ...utils import is_cython_available, requires_backends
def preprocess_item(item, keep_features=True):
    requires_backends(preprocess_item, ['cython'])
    if keep_features and 'edge_attr' in item.keys():
        edge_attr = np.asarray(item['edge_attr'], dtype=np.int64)
    else:
        edge_attr = np.ones((len(item['edge_index'][0]), 1), dtype=np.int64)
    if keep_features and 'node_feat' in item.keys():
        node_feature = np.asarray(item['node_feat'], dtype=np.int64)
    else:
        node_feature = np.ones((item['num_nodes'], 1), dtype=np.int64)
    edge_index = np.asarray(item['edge_index'], dtype=np.int64)
    input_nodes = convert_to_single_emb(node_feature) + 1
    num_nodes = item['num_nodes']
    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = np.zeros([num_nodes, num_nodes, edge_attr.shape[-1]], dtype=np.int64)
    attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_emb(edge_attr) + 1
    adj = np.zeros([num_nodes, num_nodes], dtype=bool)
    adj[edge_index[0], edge_index[1]] = True
    shortest_path_result, path = algos_graphormer.floyd_warshall(adj)
    max_dist = np.amax(shortest_path_result)
    input_edges = algos_graphormer.gen_edge_input(max_dist, path, attn_edge_type)
    attn_bias = np.zeros([num_nodes + 1, num_nodes + 1], dtype=np.single)
    item['input_nodes'] = input_nodes + 1
    item['attn_bias'] = attn_bias
    item['attn_edge_type'] = attn_edge_type
    item['spatial_pos'] = shortest_path_result.astype(np.int64) + 1
    item['in_degree'] = np.sum(adj, axis=1).reshape(-1) + 1
    item['out_degree'] = item['in_degree']
    item['input_edges'] = input_edges + 1
    if 'labels' not in item:
        item['labels'] = item['y']
    return item
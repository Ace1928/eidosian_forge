import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def test_local_subgraph_difference():
    paths = [(11, 12, 13, 14, 11, 13, 14, 12), (21, 22, 23, 24, 21, 23, 24, 22), (11, 101, 21), (12, 102, 22), (13, 103, 23), (14, 104, 24)]
    G = nx.Graph(it.chain(*[pairwise(path) for path in paths]))
    aux_graph = EdgeComponentAuxGraph.construct(G)
    subgraph_ccs = fset(aux_graph.k_edge_subgraphs(3))
    subgraph_target = fset([{101}, {102}, {103}, {104}, {21, 22, 23, 24}, {11, 12, 13, 14}])
    assert subgraph_ccs == subgraph_target
    local_ccs = fset(aux_graph.k_edge_components(3))
    local_target = fset([{101}, {102}, {103}, {104}, {11, 12, 13, 14, 21, 22, 23, 24}])
    assert local_ccs == local_target
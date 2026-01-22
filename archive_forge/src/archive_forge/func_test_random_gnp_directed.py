import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def test_random_gnp_directed():
    seeds = [21]
    for seed in seeds:
        G = nx.gnp_random_graph(20, 0.2, directed=True, seed=seed)
        _check_edge_connectivity(G)
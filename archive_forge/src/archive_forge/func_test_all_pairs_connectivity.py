import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
def test_all_pairs_connectivity(self):
    G = nx.Graph()
    nodes = [0, 1, 2, 3]
    nx.add_path(G, nodes)
    A = {n: {} for n in G}
    for u, v in itertools.combinations(nodes, 2):
        A[u][v] = A[v][u] = nx.node_connectivity(G, u, v)
    C = nx.all_pairs_node_connectivity(G)
    assert sorted(((k, sorted(v)) for k, v in A.items())) == sorted(((k, sorted(v)) for k, v in C.items()))
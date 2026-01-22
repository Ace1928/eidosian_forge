import pytest
import networkx as nx
def test_degree_k4_nodes(self):
    G = nx.complete_graph(4)
    answer = {1: 3.0, 2: 3.0}
    nd = nx.average_neighbor_degree(G, nodes=[1, 2])
    assert nd == answer
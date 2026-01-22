import pytest
import networkx as nx
def test_k_random_intersection_graph(self):
    G = nx.k_random_intersection_graph(10, 5, 2)
    assert len(G) == 10
import pytest
import networkx as nx
def test_random_intersection_graph(self):
    G = nx.uniform_random_intersection_graph(10, 5, 0.5)
    assert len(G) == 10
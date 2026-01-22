import pytest
import networkx as nx
def test_predecessor_cutoff(self):
    G = nx.path_graph(4)
    p = nx.predecessor(G, 0, 3)
    assert 4 not in p
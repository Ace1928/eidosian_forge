import pytest
import networkx as nx
def test_has_path(self):
    G = nx.Graph()
    nx.add_path(G, range(3))
    nx.add_path(G, range(3, 5))
    assert nx.has_path(G, 0, 2)
    assert not nx.has_path(G, 0, 4)
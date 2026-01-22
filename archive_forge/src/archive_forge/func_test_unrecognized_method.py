from math import sqrt
import pytest
import networkx as nx
def test_unrecognized_method(self):
    G = nx.path_graph(4)
    pytest.raises(nx.NetworkXError, nx.spectral_ordering, G, method='unknown')
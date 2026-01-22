import pytest
import networkx as nx
def test_reverse1():
    G1 = nx.Graph()
    pytest.raises(nx.NetworkXError, nx.reverse, G1)
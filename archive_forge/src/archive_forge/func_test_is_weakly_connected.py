import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_is_weakly_connected(self):
    for G, C in self.gc:
        U = G.to_undirected()
        assert nx.is_weakly_connected(G) == nx.is_connected(U)
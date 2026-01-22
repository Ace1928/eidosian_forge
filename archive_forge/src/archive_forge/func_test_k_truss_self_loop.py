import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_k_truss_self_loop(self):
    G = nx.cycle_graph(3)
    G.add_edge(0, 0)
    with pytest.raises(nx.NetworkXNotImplemented, match='Input graph has self loops'):
        nx.k_truss(G, k=1)
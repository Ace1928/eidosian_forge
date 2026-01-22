import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_core_number_self_loop(self):
    G = nx.cycle_graph(3)
    G.add_edge(0, 0)
    with pytest.raises(nx.NetworkXError, match='Input graph has self loops'):
        nx.core_number(G)
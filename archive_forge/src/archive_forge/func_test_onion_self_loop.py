import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_onion_self_loop(self):
    G = nx.cycle_graph(3)
    G.add_edge(0, 0)
    with pytest.raises(nx.NetworkXError, match='Input graph contains self loops'):
        nx.onion_layers(G)
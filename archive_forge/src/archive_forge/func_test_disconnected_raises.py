import pytest
import networkx as nx
def test_disconnected_raises():
    G = nx.ladder_graph(3)
    G.add_node('a')
    with pytest.raises(nx.NetworkXError, match='.*not connected'):
        nx.bipartite.maximal_extendability(G)
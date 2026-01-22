import pytest
import networkx as nx
def test_residual_graph_not_strongly_connected_raises():
    G = nx.Graph([(1, 2), (2, 3), (3, 4)])
    with pytest.raises(nx.NetworkXError, match='The residual graph of G is not strongly connected'):
        nx.bipartite.maximal_extendability(G)
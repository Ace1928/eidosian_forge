import pytest
import networkx as nx
def test_windmill_graph():
    for n in range(2, 20, 3):
        for k in range(2, 20, 3):
            G = nx.windmill_graph(n, k)
            assert G.number_of_nodes() == (k - 1) * n + 1
            assert G.number_of_edges() == n * k * (k - 1) / 2
            assert G.degree(0) == G.number_of_nodes() - 1
            for i in range(1, G.number_of_nodes()):
                assert G.degree(i) == k - 1
    with pytest.raises(nx.NetworkXError, match='A windmill graph must have at least two cliques'):
        nx.windmill_graph(1, 3)
    with pytest.raises(nx.NetworkXError, match='The cliques must have at least two nodes'):
        nx.windmill_graph(3, 0)
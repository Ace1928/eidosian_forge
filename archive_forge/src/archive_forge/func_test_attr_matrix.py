import pytest
import networkx as nx
def test_attr_matrix():
    G = nx.Graph()
    G.add_edge(0, 1, thickness=1, weight=3)
    G.add_edge(0, 1, thickness=1, weight=3)
    G.add_edge(0, 2, thickness=2)
    G.add_edge(1, 2, thickness=3)

    def node_attr(u):
        return G.nodes[u].get('size', 0.5) * 3

    def edge_attr(u, v):
        return G[u][v].get('thickness', 0.5)
    M = nx.attr_matrix(G, edge_attr=edge_attr, node_attr=node_attr)
    np.testing.assert_equal(M[0], np.array([[6.0]]))
    assert M[1] == [1.5]
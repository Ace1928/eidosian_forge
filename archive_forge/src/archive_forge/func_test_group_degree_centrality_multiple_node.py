import pytest
import networkx as nx
def test_group_degree_centrality_multiple_node(self):
    """
        Group degree centrality for group with more than
        1 node
        """
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
    G.add_edges_from([(1, 2), (1, 3), (1, 6), (1, 7), (1, 8), (2, 3), (2, 4), (2, 5)])
    d = nx.group_degree_centrality(G, [1, 2])
    d_answer = 1
    assert d == d_answer
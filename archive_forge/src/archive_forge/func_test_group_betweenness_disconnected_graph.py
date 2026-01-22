import pytest
import networkx as nx
def test_group_betweenness_disconnected_graph(self):
    """
        Group betweenness centrality in a disconnected graph
        """
    G = nx.path_graph(5)
    G.remove_edge(0, 1)
    C = [1]
    b = nx.group_betweenness_centrality(G, C, weight=None, normalized=False)
    b_answer = 0.0
    assert b == b_answer
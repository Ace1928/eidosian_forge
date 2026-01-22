import pytest
import networkx as nx
def test_group_betweenness_single_node(self):
    """
        Group betweenness centrality for single node group
        """
    G = nx.path_graph(5)
    C = [1]
    b = nx.group_betweenness_centrality(G, C, weight=None, normalized=False, endpoints=False)
    b_answer = 3.0
    assert b == b_answer
import pytest
import networkx as nx
def test_group_closeness_disconnected(self):
    """
        Group closeness centrality for a disconnected graph
        """
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4])
    c = nx.group_closeness_centrality(G, [1, 2])
    c_answer = 0
    assert c == c_answer
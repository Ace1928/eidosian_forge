import pytest
import networkx as nx
def test_group_closeness_single_node(self):
    """
        Group closeness centrality for a single node group
        """
    G = nx.path_graph(5)
    c = nx.group_closeness_centrality(G, [1])
    c_answer = nx.closeness_centrality(G, 1)
    assert c == c_answer
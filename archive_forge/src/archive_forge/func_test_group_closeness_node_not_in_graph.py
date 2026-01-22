import pytest
import networkx as nx
def test_group_closeness_node_not_in_graph(self):
    """
        Node(s) in S not in graph, raises NodeNotFound exception
        """
    with pytest.raises(nx.NodeNotFound):
        nx.group_closeness_centrality(nx.path_graph(5), [6, 7, 8])
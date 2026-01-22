import networkx as nx
def test_disconnecting_graph(self):
    """Tests that the closeness vitality of a node whose removal
        disconnects the graph is negative infinity.

        """
    G = nx.path_graph(3)
    assert nx.closeness_vitality(G, node=1) == -float('inf')
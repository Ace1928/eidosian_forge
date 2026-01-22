import networkx as nx
def test_local_efficiency_complete_graph(self):
    """
        Test that the local efficiency for a complete graph with at least 3
        nodes should be one. For a graph with only 2 nodes, the induced
        subgraph has no edges.
        """
    for n in range(3, 10):
        G = nx.complete_graph(n)
        assert nx.local_efficiency(G) == 1
import networkx as nx
def test_global_efficiency_complete_graph(self):
    """
        Tests that the average global efficiency of the complete graph is one.
        """
    for n in range(2, 10):
        G = nx.complete_graph(n)
        assert nx.global_efficiency(G) == 1
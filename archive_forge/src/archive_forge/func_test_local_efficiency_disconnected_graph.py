import networkx as nx
def test_local_efficiency_disconnected_graph(self):
    """
        In a disconnected graph the efficiency is 0
        """
    assert nx.local_efficiency(self.G1) == 0
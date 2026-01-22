import networkx as nx
def test_using_ego_graph(self):
    """
        Test that the ego graph is used when computing local efficiency.
        For more information, see GitHub issue #2710.
        """
    assert nx.local_efficiency(self.G3) == 7 / 12
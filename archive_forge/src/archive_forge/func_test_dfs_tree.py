import networkx as nx
def test_dfs_tree(self):
    exp_nodes = sorted(self.G.nodes())
    exp_edges = [(0, 1), (1, 2), (1, 3), (2, 4)]
    T = nx.dfs_tree(self.G, source=0)
    assert sorted(T.nodes()) == exp_nodes
    assert sorted(T.edges()) == exp_edges
    T = nx.dfs_tree(self.G, source=None)
    assert sorted(T.nodes()) == exp_nodes
    assert sorted(T.edges()) == exp_edges
    T = nx.dfs_tree(self.G)
    assert sorted(T.nodes()) == exp_nodes
    assert sorted(T.edges()) == exp_edges
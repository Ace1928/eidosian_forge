import networkx as nx
def test_dls_tree(self):
    T = nx.dfs_tree(self.G, source=3, depth_limit=1)
    assert sorted(T.edges()) == [(3, 2), (3, 4)]
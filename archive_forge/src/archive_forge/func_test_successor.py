import networkx as nx
def test_successor(self):
    assert nx.dfs_successors(self.G, source=0) == {0: [1], 1: [2, 3], 2: [4]}
    assert nx.dfs_successors(self.G, source=1) == {0: [3, 4], 1: [0], 4: [2]}
    assert nx.dfs_successors(self.D) == {0: [1], 2: [3]}
    assert nx.dfs_successors(self.D, source=1) == {1: [0]}
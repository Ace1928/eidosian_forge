import networkx as nx
def test_dls_successor(self):
    result = nx.dfs_successors(self.G, source=4, depth_limit=3)
    assert {n: set(v) for n, v in result.items()} == {2: {1, 7}, 3: {2}, 4: {3, 5}, 5: {6}}
    result = nx.dfs_successors(self.D, source=7, depth_limit=2)
    assert {n: set(v) for n, v in result.items()} == {8: {9}, 2: {3}, 7: {8, 2}}
import networkx as nx
def test_dfs_labeled_edges(self):
    edges = list(nx.dfs_labeled_edges(self.G, source=0))
    forward = [(u, v) for u, v, d in edges if d == 'forward']
    assert forward == [(0, 0), (0, 1), (1, 2), (2, 4), (1, 3)]
    assert edges == [(0, 0, 'forward'), (0, 1, 'forward'), (1, 0, 'nontree'), (1, 2, 'forward'), (2, 1, 'nontree'), (2, 4, 'forward'), (4, 2, 'nontree'), (4, 0, 'nontree'), (2, 4, 'reverse'), (1, 2, 'reverse'), (1, 3, 'forward'), (3, 1, 'nontree'), (3, 0, 'nontree'), (1, 3, 'reverse'), (0, 1, 'reverse'), (0, 3, 'nontree'), (0, 4, 'nontree'), (0, 0, 'reverse')]
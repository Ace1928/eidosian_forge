from functools import partial
import pytest
import networkx as nx
def test_limited_bfs_layers(self):
    assert dict(enumerate(nx.bfs_layers(self.G, sources=[0]))) == {0: [0], 1: [1], 2: [2], 3: [3, 7], 4: [4, 8], 5: [5, 9], 6: [6, 10]}
    assert dict(enumerate(nx.bfs_layers(self.D, sources=2))) == {0: [2], 1: [3, 7], 2: [8], 3: [9], 4: [10]}
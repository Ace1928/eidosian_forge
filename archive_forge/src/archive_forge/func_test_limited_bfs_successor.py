from functools import partial
import pytest
import networkx as nx
def test_limited_bfs_successor(self):
    assert dict(nx.bfs_successors(self.G, source=1, depth_limit=3)) == {1: [0, 2], 2: [3, 7], 3: [4], 7: [8]}
    result = {n: sorted(s) for n, s in nx.bfs_successors(self.D, source=7, depth_limit=2)}
    assert result == {8: [9], 2: [3], 7: [2, 8]}
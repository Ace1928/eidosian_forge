from functools import partial
import pytest
import networkx as nx
def test_limited_bfs_predecessor(self):
    assert dict(nx.bfs_predecessors(self.G, source=1, depth_limit=3)) == {0: 1, 2: 1, 3: 2, 4: 3, 7: 2, 8: 7}
    assert dict(nx.bfs_predecessors(self.D, source=7, depth_limit=2)) == {2: 7, 3: 2, 8: 7, 9: 8}
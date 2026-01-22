import pytest
import networkx as nx
def test_no_weight(self):
    inf = float('inf')
    expected = {(3, 4, inf), (4, 3, inf)}
    assert next(nx.local_bridges(self.BB)) in expected
    expected = {(u, v, 3) for u, v in self.square.edges}
    assert set(nx.local_bridges(self.square)) == expected
    assert list(nx.local_bridges(self.tri)) == []
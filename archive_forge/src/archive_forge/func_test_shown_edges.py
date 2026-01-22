import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_shown_edges(self):
    show_edges = [(2, 3, 4), (2, 3, 3), (8, 7, 0), (222, 223, 0)]
    edge_subgraph = self.show_edges_filter(show_edges)
    G = self.gview(self.G, filter_edge=edge_subgraph)
    assert self.G.nodes == G.nodes
    if G.is_directed():
        assert G.edges == {(2, 3, 4)}
        assert list(G[3]) == []
        assert list(G.pred[3]) == [2]
        assert list(G.pred[2]) == []
        assert G.size() == 1
    else:
        assert G.edges == {(2, 3, 4), (7, 8, 0)}
        assert G.size() == 2
        assert list(G[3]) == [2]
    assert G.degree(3) == 1
    assert list(G[2]) == [3]
    pytest.raises(KeyError, G.__getitem__, 221)
    pytest.raises(KeyError, G.__getitem__, 222)
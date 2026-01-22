import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_hidden_nodes(self):
    hide_nodes = [4, 5, 111]
    nodes_gone = nx.filters.hide_nodes(hide_nodes)
    gview = self.gview
    G = gview(self.G, filter_node=nodes_gone)
    assert self.G.nodes - G.nodes == {4, 5}
    assert self.G.edges - G.edges == self.hide_edges_w_hide_nodes
    if G.is_directed():
        assert list(G[3]) == []
        assert list(G[2]) == [3]
    else:
        assert list(G[3]) == [2]
        assert set(G[2]) == {1, 3}
    pytest.raises(KeyError, G.__getitem__, 4)
    pytest.raises(KeyError, G.__getitem__, 112)
    pytest.raises(KeyError, G.__getitem__, 111)
    assert G.degree(3) == (3 if G.is_multigraph() else 1)
    assert G.size() == (7 if G.is_multigraph() else 5)
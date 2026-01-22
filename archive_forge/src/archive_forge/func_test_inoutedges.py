import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_inoutedges(self):
    edges_gone = self.hide_edges_filter(self.hide_edges)
    hide_nodes = [4, 5, 111]
    nodes_gone = nx.filters.hide_nodes(hide_nodes)
    G = self.gview(self.G, filter_node=nodes_gone, filter_edge=edges_gone)
    assert self.G.in_edges - G.in_edges == self.excluded
    assert self.G.out_edges - G.out_edges == self.excluded
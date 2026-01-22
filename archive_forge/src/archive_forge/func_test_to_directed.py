import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_to_directed(self):
    G = self.G()
    if not G.is_directed():
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'D')])
        DG = G.to_directed()
        assert DG is not G
        assert DG.is_directed()
        assert DG.name == G.name
        assert DG.adj == G.adj
        assert sorted(DG.out_edges(list('AB'))) == [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('B', 'D')]
        DG.remove_edge('A', 'B')
        assert DG.has_edge('B', 'A')
        assert not DG.has_edge('A', 'B')
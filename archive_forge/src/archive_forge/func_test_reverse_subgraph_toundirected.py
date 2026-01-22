import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_reverse_subgraph_toundirected(self):
    G = self.DG.reverse(copy=False)
    SG = G.subgraph([4, 5, 6])
    SSG = SG.to_undirected()
    assert list(SSG) == [4, 5, 6]
    assert sorted(SSG.edges) == [(4, 5), (5, 6)]
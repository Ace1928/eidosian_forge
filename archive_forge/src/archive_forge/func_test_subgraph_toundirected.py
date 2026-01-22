import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_subgraph_toundirected(self):
    SG = nx.induced_subgraph(self.G, [4, 5, 6])
    SSG = SG.to_undirected()
    assert list(SSG) == [4, 5, 6]
    assert sorted(SSG.edges) == [(4, 5), (5, 6)]
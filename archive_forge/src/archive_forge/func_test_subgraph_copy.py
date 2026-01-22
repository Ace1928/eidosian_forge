import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_subgraph_copy(self):
    for origG in self.graphs:
        G = nx.Graph(origG)
        SG = G.subgraph([4, 5, 6])
        H = SG.copy()
        assert type(G) == type(H)
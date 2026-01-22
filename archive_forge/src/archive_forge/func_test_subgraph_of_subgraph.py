import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_subgraph_of_subgraph(self):
    SGv = nx.subgraph(self.G, range(3, 7))
    SDGv = nx.subgraph(self.DG, range(3, 7))
    SMGv = nx.subgraph(self.MG, range(3, 7))
    SMDGv = nx.subgraph(self.MDG, range(3, 7))
    for G in self.graphs + [SGv, SDGv, SMGv, SMDGv]:
        SG = nx.induced_subgraph(G, [4, 5, 6])
        assert list(SG) == [4, 5, 6]
        SSG = SG.subgraph([6, 7])
        assert list(SSG) == [6]
        assert SSG._graph is G
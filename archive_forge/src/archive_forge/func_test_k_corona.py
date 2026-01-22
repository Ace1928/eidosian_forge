import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_k_corona(self):
    k_corona_subgraph = nx.k_corona(self.H, k=2)
    assert sorted(k_corona_subgraph.nodes()) == [2, 4, 5, 6]
    k_corona_subgraph = nx.k_corona(self.H, k=1)
    assert sorted(k_corona_subgraph.nodes()) == [1]
    k_corona_subgraph = nx.k_corona(self.H, k=0)
    assert sorted(k_corona_subgraph.nodes()) == [0]
import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_k_core(self):
    k_core_subgraph = nx.k_core(self.H, k=0)
    assert sorted(k_core_subgraph.nodes()) == sorted(self.H.nodes())
    k_core_subgraph = nx.k_core(self.H, k=1)
    assert sorted(k_core_subgraph.nodes()) == [1, 2, 3, 4, 5, 6]
    k_core_subgraph = nx.k_core(self.H, k=2)
    assert sorted(k_core_subgraph.nodes()) == [2, 4, 5, 6]
import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_k_shell(self):
    k_shell_subgraph = nx.k_shell(self.H, k=2)
    assert sorted(k_shell_subgraph.nodes()) == [2, 4, 5, 6]
    k_shell_subgraph = nx.k_shell(self.H, k=1)
    assert sorted(k_shell_subgraph.nodes()) == [1, 3]
    k_shell_subgraph = nx.k_shell(self.H, k=0)
    assert sorted(k_shell_subgraph.nodes()) == [0]
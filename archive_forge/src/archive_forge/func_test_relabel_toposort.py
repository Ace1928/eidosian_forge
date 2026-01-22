import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_toposort(self):
    K5 = nx.complete_graph(4)
    G = nx.complete_graph(4)
    G = nx.relabel_nodes(G, {i: i + 1 for i in range(4)}, copy=False)
    assert nx.is_isomorphic(K5, G)
    G = nx.complete_graph(4)
    G = nx.relabel_nodes(G, {i: i - 1 for i in range(4)}, copy=False)
    assert nx.is_isomorphic(K5, G)
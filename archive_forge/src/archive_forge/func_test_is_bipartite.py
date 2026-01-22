import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_is_bipartite(self):
    assert bipartite.is_bipartite(nx.path_graph(4))
    assert bipartite.is_bipartite(nx.DiGraph([(1, 0)]))
    assert not bipartite.is_bipartite(nx.complete_graph(3))
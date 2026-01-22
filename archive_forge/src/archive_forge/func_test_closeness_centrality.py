import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_closeness_centrality(self):
    c = bipartite.closeness_centrality(self.P4, [1, 3])
    answer = {0: 2.0 / 3, 1: 1.0, 2: 1.0, 3: 2.0 / 3}
    assert c == answer
    c = bipartite.closeness_centrality(self.K3, [0, 1, 2])
    answer = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
    assert c == answer
    c = bipartite.closeness_centrality(self.C4, [0, 2])
    answer = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    assert c == answer
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    c = bipartite.closeness_centrality(G, [0])
    assert c == {0: 0.0, 1: 0.0}
    c = bipartite.closeness_centrality(G, [1])
    assert c == {0: 0.0, 1: 0.0}
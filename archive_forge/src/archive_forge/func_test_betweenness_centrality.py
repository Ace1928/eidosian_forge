import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_betweenness_centrality(self):
    c = bipartite.betweenness_centrality(self.P4, [1, 3])
    answer = {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0}
    assert c == answer
    c = bipartite.betweenness_centrality(self.K3, [0, 1, 2])
    answer = {0: 0.125, 1: 0.125, 2: 0.125, 3: 0.125, 4: 0.125, 5: 0.125}
    assert c == answer
    c = bipartite.betweenness_centrality(self.C4, [0, 2])
    answer = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    assert c == answer
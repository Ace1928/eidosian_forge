import pytest
import networkx as nx
import networkx.algorithms.threshold as nxt
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test_find_alternating_4_cycle(self):
    G = nx.Graph()
    G.add_edge(1, 2)
    assert not nxt.find_alternating_4_cycle(G)
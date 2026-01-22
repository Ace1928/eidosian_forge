import pytest
import networkx as nx
import networkx.algorithms.threshold as nxt
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test_threshold_sequence_graph_test(self):
    G = nx.star_graph(10)
    assert nxt.is_threshold_graph(G)
    assert nxt.is_threshold_sequence([d for n, d in G.degree()])
    G = nx.complete_graph(10)
    assert nxt.is_threshold_graph(G)
    assert nxt.is_threshold_sequence([d for n, d in G.degree()])
    deg = [3, 2, 2, 1, 1, 1]
    assert not nxt.is_threshold_sequence(deg)
    deg = [3, 2, 2, 1]
    assert nxt.is_threshold_sequence(deg)
    G = nx.generators.havel_hakimi_graph(deg)
    assert nxt.is_threshold_graph(G)
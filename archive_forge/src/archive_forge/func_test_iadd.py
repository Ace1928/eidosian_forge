import random
import pytest
import cirq.contrib.graph_device as ccgd
def test_iadd():
    graph = ccgd.UndirectedHypergraph(labelled_edges={(0, 1): None})
    addend = ccgd.UndirectedHypergraph(labelled_edges={(1, 2): None})
    graph += addend
    assert set(graph.edges) == set((frozenset(e) for e in ((0, 1), (1, 2))))
    assert sorted(graph.vertices) == [0, 1, 2]
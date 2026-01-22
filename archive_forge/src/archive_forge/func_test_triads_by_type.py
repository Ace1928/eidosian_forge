import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
def test_triads_by_type():
    """Tests the all_triplets function."""
    G = nx.DiGraph()
    G.add_edges_from(['01', '02', '03', '04', '05', '12', '16', '51', '56', '65'])
    all_triads = nx.all_triads(G)
    expected = defaultdict(list)
    for triad in all_triads:
        name = nx.triad_type(triad)
        expected[name].append(triad)
    actual = nx.triads_by_type(G)
    assert set(actual.keys()) == set(expected.keys())
    for tri_type, actual_Gs in actual.items():
        expected_Gs = expected[tri_type]
        for a in actual_Gs:
            assert any((nx.is_isomorphic(a, e) for e in expected_Gs))
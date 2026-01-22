from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_petersen_graph(self):
    G = nx.petersen_graph()
    mcb = list(nx.minimum_cycle_basis(G))
    expected = [[4, 9, 7, 5, 0], [1, 2, 3, 4, 0], [1, 6, 8, 5, 0], [4, 3, 8, 5, 0], [1, 6, 9, 4, 0], [1, 2, 7, 5, 0]]
    assert len(mcb) == len(expected)
    assert all((c in expected for c in mcb))
    for c in mcb:
        assert all((G.has_edge(u, v) for u, v in nx.utils.pairwise(c, cyclic=True)))
    check_independent(mcb)
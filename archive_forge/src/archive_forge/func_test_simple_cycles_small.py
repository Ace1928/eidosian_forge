from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_simple_cycles_small(self):
    G = nx.DiGraph()
    nx.add_cycle(G, [1, 2, 3])
    c = sorted(nx.simple_cycles(G))
    assert len(c) == 1
    assert self.is_cyclic_permutation(c[0], [1, 2, 3])
    nx.add_cycle(G, [10, 20, 30])
    cc = sorted(nx.simple_cycles(G))
    assert len(cc) == 2
    ca = [[1, 2, 3], [10, 20, 30]]
    for c in cc:
        assert any((self.is_cyclic_permutation(c, rc) for rc in ca))
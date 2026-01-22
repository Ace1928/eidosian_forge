from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_cycle_basis_ordered(self):
    G = nx.cycle_graph(5)
    G.update(nx.cycle_graph(range(3, 8)))
    cbG = nx.cycle_basis(G)
    perm = {1: 0, 0: 1}
    H = nx.relabel_nodes(G, perm)
    cbH = [[perm.get(n, n) for n in cyc] for cyc in nx.cycle_basis(H)]
    assert cbG == cbH
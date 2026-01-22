import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
def test_const_covered_neighbors(self):
    G1 = nx.DiGraph([(0, 1), (1, 2), (0, 3), (2, 3)])
    G2 = nx.DiGraph([('a', 'b'), ('b', 'c'), ('a', 'k'), ('c', 'k')])
    gparams = _GraphParameters(G1, G2, None, None, None, None, None)
    sparams = _StateParameters({0: 'a', 1: 'b', 2: 'c'}, {'a': 0, 'b': 1, 'c': 2}, None, None, None, None, None, None, None, None)
    u, v = (3, 'k')
    assert _consistent_PT(u, v, gparams, sparams)
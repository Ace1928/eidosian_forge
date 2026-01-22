import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_different_order1(self):
    G1 = nx.path_graph(5)
    G2 = nx.path_graph(6)
    assert not vf2pp_is_isomorphic(G1, G2)
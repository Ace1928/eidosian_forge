import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_different_order2(self):
    G1 = nx.barbell_graph(100, 20)
    G2 = nx.barbell_graph(101, 20)
    assert not vf2pp_is_isomorphic(G1, G2)
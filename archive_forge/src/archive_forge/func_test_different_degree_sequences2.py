import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_different_degree_sequences2(self):
    G1 = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 3), (4, 7), (7, 8), (8, 3)])
    G2 = G1.copy()
    G2.add_edge(8, 0)
    assert not vf2pp_is_isomorphic(G1, G2)
    G1.add_edge(6, 1)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(['a']))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle('a'))), 'label')
    assert vf2pp_is_isomorphic(G1, G2)
import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_different_degree_sequences3(self):
    G1 = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (2, 5), (2, 6)])
    G2 = nx.Graph([(0, 1), (0, 6), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (2, 5), (2, 6)])
    assert not vf2pp_is_isomorphic(G1, G2)
    G1.add_edge(3, 5)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(['a']))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle('a'))), 'label')
    assert vf2pp_is_isomorphic(G1, G2)
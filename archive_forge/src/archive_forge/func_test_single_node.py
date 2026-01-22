import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
def test_single_node(self):
    G1 = nx.Graph()
    G2 = nx.Graph()
    G1.add_node(1)
    G2.add_node(1)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_many))), 'label')
    l1, l2 = (nx.get_node_attributes(G1, 'label'), nx.get_node_attributes(G2, 'label'))
    gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), nx.utils.groups(dict(G2.degree())))
    m = _matching_order(gparams)
    assert m == [1]
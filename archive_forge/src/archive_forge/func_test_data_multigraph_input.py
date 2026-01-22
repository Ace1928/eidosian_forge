from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
def test_data_multigraph_input(self):
    edata0 = {'w': 200, 's': 'foo'}
    edata1 = {'w': 201, 's': 'bar'}
    keydict = {0: edata0, 1: edata1}
    dododod = {'a': {'b': keydict}}
    multiple_edge = [('a', 'b', 0, edata0), ('a', 'b', 1, edata1)]
    single_edge = [('a', 'b', 0, keydict)]
    G = self.Graph(dododod, multigraph_input=True)
    assert list(G.edges(keys=True, data=True)) == multiple_edge
    G = self.Graph(dododod, multigraph_input=None)
    assert list(G.edges(keys=True, data=True)) == multiple_edge
    G = self.Graph(dododod, multigraph_input=False)
    assert list(G.edges(keys=True, data=True)) == single_edge
    G = self.Graph(dododod, multigraph_input=True)
    H = self.Graph(nx.to_dict_of_dicts(G))
    assert nx.is_isomorphic(G, H) is True
    for mgi in [True, False]:
        H = self.Graph(nx.to_dict_of_dicts(G), multigraph_input=mgi)
        assert nx.is_isomorphic(G, H) == mgi
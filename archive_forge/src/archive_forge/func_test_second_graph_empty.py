import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
@pytest.mark.parametrize('graph_type', (nx.Graph, nx.MultiGraph, nx.DiGraph))
def test_second_graph_empty(self, graph_type):
    G = graph_type([(0, 1)])
    H = graph_type()
    assert vf2pp_isomorphism(G, H) is None
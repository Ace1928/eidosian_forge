import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_invalid_edge(self):
    G = nx.path_graph(4)
    assert not nx.is_matching(G, {(0, 3), (1, 2)})
    raises(nx.NetworkXError, nx.is_matching, G, {(0, 55)})
    G = nx.DiGraph(G.edges)
    assert nx.is_matching(G, {(0, 1)})
    assert not nx.is_matching(G, {(1, 0)})
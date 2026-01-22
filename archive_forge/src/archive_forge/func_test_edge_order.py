import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_edge_order(self):
    G = nx.path_graph(4)
    assert nx.is_matching(G, {(0, 1), (2, 3)})
    assert nx.is_matching(G, {(1, 0), (2, 3)})
    assert nx.is_matching(G, {(0, 1), (3, 2)})
    assert nx.is_matching(G, {(1, 0), (3, 2)})
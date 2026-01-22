import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_valid_matching(self):
    edges = [(1, 2), (1, 5), (2, 3), (2, 5), (3, 4), (3, 6), (5, 6)]
    G = nx.Graph(edges)
    matching = nx.maximal_matching(G)
    assert nx.is_maximal_matching(G, matching)
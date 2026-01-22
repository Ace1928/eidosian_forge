import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_nasty_blossom_augmenting(self):
    """Create nested blossom, relabel as T in more than one way"""
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 45), (1, 7, 45), (2, 3, 50), (3, 4, 45), (4, 5, 95), (4, 6, 94), (5, 6, 94), (6, 7, 50), (1, 8, 30), (3, 11, 35), (5, 9, 36), (7, 10, 26), (11, 12, 5)])
    ans = {1: 8, 2: 3, 3: 2, 4: 6, 5: 9, 6: 4, 7: 10, 8: 1, 9: 5, 10: 7, 11: 12, 12: 11}
    answer = matching_dict_to_set(ans)
    assert edges_equal(nx.max_weight_matching(G), answer)
    assert edges_equal(nx.min_weight_matching(G), answer)
import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_nasty_blossom_least_slack(self):
    """Create blossom, relabel as T, expand such that a new
        least-slack S-to-free dge is produced, augment:
        """
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 45), (1, 5, 45), (2, 3, 50), (3, 4, 45), (4, 5, 50), (1, 6, 30), (3, 9, 35), (4, 8, 28), (5, 7, 26), (9, 10, 5)])
    ans = {1: 6, 2: 3, 3: 2, 4: 8, 5: 7, 6: 1, 7: 5, 8: 4, 9: 10, 10: 9}
    answer = matching_dict_to_set(ans)
    assert edges_equal(nx.max_weight_matching(G), answer)
    assert edges_equal(nx.min_weight_matching(G), answer)
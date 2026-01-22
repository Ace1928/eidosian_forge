import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_edge_attribute_name(self):
    G = nx.Graph()
    G.add_edge('one', 'two', weight=10, abcd=11)
    G.add_edge('two', 'three', weight=11, abcd=10)
    assert edges_equal(nx.max_weight_matching(G, weight='abcd'), matching_dict_to_set({'one': 'two', 'two': 'one'}))
    assert edges_equal(nx.min_weight_matching(G, weight='abcd'), matching_dict_to_set({'three': 'two'}))
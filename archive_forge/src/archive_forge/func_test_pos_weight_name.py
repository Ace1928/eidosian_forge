import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_pos_weight_name(self):
    gtg = nx.geographical_threshold_graph
    G = gtg(50, 100, seed=42, pos_name='coords', weight_name='wt')
    assert all((len(d['coords']) == 2 for n, d in G.nodes.items()))
    assert all((d['wt'] > 0 for n, d in G.nodes.items()))
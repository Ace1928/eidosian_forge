from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_multigraph_weighted(self):
    G = nx.MultiDiGraph()
    edges = [(1, 2, 2), (2, 3, 2), (1, 3, 1), (1, 3, 5), (1, 3, 2)]
    G.add_weighted_edges_from(edges)
    assert nx.dag_longest_path_length(G) == 5
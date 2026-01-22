from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_unweighted2(self):
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 3), (1, 5), (3, 5)]
    G = nx.DiGraph(edges)
    assert nx.dag_longest_path(G) == [1, 2, 3, 4, 5]
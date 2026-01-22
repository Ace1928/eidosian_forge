from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_unweighted1(self):
    edges = [(1, 2), (2, 3), (2, 4), (3, 5), (5, 6), (3, 7)]
    G = nx.DiGraph(edges)
    assert nx.dag_longest_path(G) == [1, 2, 3, 5, 6]
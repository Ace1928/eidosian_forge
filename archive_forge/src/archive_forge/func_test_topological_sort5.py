from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_topological_sort5(self):
    G = nx.DiGraph()
    G.add_edge(0, 1)
    assert list(nx.topological_sort(G)) == [0, 1]
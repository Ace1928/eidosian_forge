from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_all_topological_sorts_1(self):
    DG = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 5)])
    assert list(nx.all_topological_sorts(DG)) == [[1, 2, 3, 4, 5]]
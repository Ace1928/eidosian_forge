from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_all_topological_sorts_4(self):
    DG = nx.DiGraph()
    for i in range(7):
        DG.add_node(i)
    assert sorted(map(list, permutations(DG.nodes))) == sorted(nx.all_topological_sorts(DG))
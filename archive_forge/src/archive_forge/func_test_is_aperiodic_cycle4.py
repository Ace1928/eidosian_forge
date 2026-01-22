from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_is_aperiodic_cycle4():
    G = nx.DiGraph()
    nx.add_cycle(G, [1, 2, 3, 4])
    G.add_edge(1, 3)
    assert nx.is_aperiodic(G)
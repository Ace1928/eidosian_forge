from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_is_aperiodic_disconnected():
    G = nx.DiGraph()
    nx.add_cycle(G, [1, 2, 3, 4])
    nx.add_cycle(G, [5, 6, 7, 8])
    assert not nx.is_aperiodic(G)
    G.add_edge(1, 3)
    G.add_edge(5, 7)
    assert nx.is_aperiodic(G)
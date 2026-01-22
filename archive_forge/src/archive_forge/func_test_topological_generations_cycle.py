from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_topological_generations_cycle():
    G = nx.DiGraph([[2, 1], [3, 1], [1, 2]])
    with pytest.raises(nx.NetworkXUnfeasible):
        list(nx.topological_generations(G))
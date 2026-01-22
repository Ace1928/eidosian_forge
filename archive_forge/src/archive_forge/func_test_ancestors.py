from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_ancestors(self):
    G = nx.DiGraph()
    ancestors = nx.algorithms.dag.ancestors
    G.add_edges_from([(1, 2), (1, 3), (4, 2), (4, 3), (4, 5), (2, 6), (5, 6)])
    assert ancestors(G, 6) == {1, 2, 4, 5}
    assert ancestors(G, 3) == {1, 4}
    assert ancestors(G, 1) == set()
    pytest.raises(nx.NetworkXError, ancestors, G, 8)
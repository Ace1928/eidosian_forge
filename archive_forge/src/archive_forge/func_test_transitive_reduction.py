from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_transitive_reduction(self):
    G = nx.DiGraph([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])
    transitive_reduction = nx.algorithms.dag.transitive_reduction
    solution = [(1, 2), (2, 3), (3, 4)]
    assert edges_equal(transitive_reduction(G).edges(), solution)
    G = nx.DiGraph([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)])
    transitive_reduction = nx.algorithms.dag.transitive_reduction
    solution = [(1, 2), (2, 3), (2, 4)]
    assert edges_equal(transitive_reduction(G).edges(), solution)
    G = nx.Graph([(1, 2), (2, 3), (3, 4)])
    pytest.raises(nx.NetworkXNotImplemented, transitive_reduction, G)
from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_antichains(self):
    antichains = nx.algorithms.dag.antichains
    G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    solution = [[], [4], [3], [2], [1]]
    self._check_antichains(list(antichains(G)), solution)
    G = nx.DiGraph([(1, 2), (2, 3), (2, 4), (3, 5), (5, 6), (5, 7)])
    solution = [[], [4], [7], [7, 4], [6], [6, 4], [6, 7], [6, 7, 4], [5], [5, 4], [3], [3, 4], [2], [1]]
    self._check_antichains(list(antichains(G)), solution)
    G = nx.DiGraph([(1, 2), (1, 3), (3, 4), (3, 5), (5, 6)])
    solution = [[], [6], [5], [4], [4, 6], [4, 5], [3], [2], [2, 6], [2, 5], [2, 4], [2, 4, 6], [2, 4, 5], [2, 3], [1]]
    self._check_antichains(list(antichains(G)), solution)
    G = nx.DiGraph({0: [1, 2], 1: [4], 2: [3], 3: [4]})
    solution = [[], [4], [3], [2], [1], [1, 3], [1, 2], [0]]
    self._check_antichains(list(antichains(G)), solution)
    G = nx.DiGraph()
    self._check_antichains(list(antichains(G)), [[]])
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2])
    solution = [[], [0], [1], [1, 0], [2], [2, 0], [2, 1], [2, 1, 0]]
    self._check_antichains(list(antichains(G)), solution)

    def f(x):
        return list(antichains(x))
    G = nx.Graph([(1, 2), (2, 3), (3, 4)])
    pytest.raises(nx.NetworkXNotImplemented, f, G)
    G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    pytest.raises(nx.NetworkXUnfeasible, f, G)
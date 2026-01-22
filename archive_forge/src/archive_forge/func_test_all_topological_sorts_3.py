from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_all_topological_sorts_3(self):

    def unfeasible():
        DG = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 2), (4, 5)])
        list(nx.all_topological_sorts(DG))

    def not_implemented():
        G = nx.Graph([(1, 2), (2, 3)])
        list(nx.all_topological_sorts(G))

    def not_implemented_2():
        G = nx.MultiGraph([(1, 2), (1, 2), (2, 3)])
        list(nx.all_topological_sorts(G))
    pytest.raises(nx.NetworkXUnfeasible, unfeasible)
    pytest.raises(nx.NetworkXNotImplemented, not_implemented)
    pytest.raises(nx.NetworkXNotImplemented, not_implemented_2)
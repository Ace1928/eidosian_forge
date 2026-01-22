from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_topological_sort3(self):
    DG = nx.DiGraph()
    DG.add_edges_from([(1, i) for i in range(2, 5)])
    DG.add_edges_from([(2, i) for i in range(5, 9)])
    DG.add_edges_from([(6, i) for i in range(9, 12)])
    DG.add_edges_from([(4, i) for i in range(12, 15)])

    def validate(order):
        assert isinstance(order, list)
        assert set(order) == set(DG)
        for u, v in combinations(order, 2):
            assert not nx.has_path(DG, v, u)
    validate(list(nx.topological_sort(DG)))
    DG.add_edge(14, 1)
    pytest.raises(nx.NetworkXUnfeasible, _consume, nx.topological_sort(DG))
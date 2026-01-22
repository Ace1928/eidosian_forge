import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_minimum_weight_full_matching_with_no_full_matching(self):
    B = nx.Graph()
    B.add_nodes_from([1, 2, 3], bipartite=0)
    B.add_nodes_from([4, 5, 6], bipartite=1)
    B.add_edge(1, 4, weight=100)
    B.add_edge(2, 4, weight=100)
    B.add_edge(3, 4, weight=50)
    B.add_edge(3, 5, weight=50)
    B.add_edge(3, 6, weight=50)
    with pytest.raises(ValueError):
        minimum_weight_full_matching(B)
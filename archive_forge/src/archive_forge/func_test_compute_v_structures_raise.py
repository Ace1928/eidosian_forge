from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_compute_v_structures_raise():
    G = nx.Graph()
    pytest.raises(nx.NetworkXNotImplemented, nx.compute_v_structures, G)
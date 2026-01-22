from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_is_aperiodic_undirected_raises():
    G = nx.Graph()
    pytest.raises(nx.NetworkXError, nx.is_aperiodic, G)
from itertools import combinations
import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_empty_raises(self):
    with pytest.raises(nx.NetworkXError):
        G = nx.empty_graph()
        T = nx.gomory_hu_tree(G)
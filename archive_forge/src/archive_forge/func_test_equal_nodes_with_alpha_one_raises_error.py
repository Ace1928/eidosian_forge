import math
from functools import partial
import pytest
import networkx as nx
def test_equal_nodes_with_alpha_one_raises_error(self):
    G = nx.complete_graph(4)
    assert pytest.raises(nx.NetworkXAlgorithmError, self.test, G, [(0, 0)], [], alpha=1.0)
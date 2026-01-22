import pytest
import networkx
def test_invalid1(self):
    pytest.raises((TypeError, networkx.NetworkXError), networkx.random_clustered_graph, [[1, 1], [2, 1], [0, 1]])
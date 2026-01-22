import pytest
import networkx as nx
def test_general_random_intersection_graph(self):
    G = nx.general_random_intersection_graph(10, 5, [0.1, 0.2, 0.2, 0.1, 0.1])
    assert len(G) == 10
    pytest.raises(ValueError, nx.general_random_intersection_graph, 10, 5, [0.1, 0.2, 0.2, 0.1])
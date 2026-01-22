import pytest
import networkx as nx
from networkx.utils import pairwise
def test_absent_source_bellman_ford(self):
    G = nx.path_graph(2)
    for fn in (nx.bellman_ford_predecessor_and_distance, nx.bellman_ford_path, nx.bellman_ford_path_length, nx.single_source_bellman_ford_path, nx.single_source_bellman_ford_path_length, nx.single_source_bellman_ford):
        pytest.raises(nx.NodeNotFound, fn, G, 3, 0)
        pytest.raises(nx.NodeNotFound, fn, G, 3, 3)
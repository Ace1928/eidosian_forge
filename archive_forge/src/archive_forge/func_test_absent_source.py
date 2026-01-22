import pytest
import networkx as nx
from networkx.utils import pairwise
@pytest.mark.parametrize('fn', (nx.multi_source_dijkstra_path, nx.multi_source_dijkstra_path_length, nx.multi_source_dijkstra))
def test_absent_source(self, fn):
    G = nx.path_graph(2)
    with pytest.raises(nx.NodeNotFound):
        fn(G, [3], 0)
    with pytest.raises(nx.NodeNotFound):
        fn(G, [3], 3)
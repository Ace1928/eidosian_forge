import pytest
import networkx as nx
def test_single_source_all_shortest_paths_bad_method(self):
    with pytest.raises(ValueError):
        G = nx.path_graph(2)
        dict(nx.single_source_all_shortest_paths(G, 0, weight='weight', method='SPAM'))
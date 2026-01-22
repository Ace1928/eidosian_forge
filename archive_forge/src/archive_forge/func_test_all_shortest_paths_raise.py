import pytest
import networkx as nx
def test_all_shortest_paths_raise(self):
    with pytest.raises(nx.NetworkXNoPath):
        G = nx.path_graph(4)
        G.add_node(4)
        list(nx.all_shortest_paths(G, 0, 4))
import pytest
import networkx as nx
def test_self_loops_square_clustering(self):
    G = nx.path_graph(5)
    assert nx.square_clustering(G) == {0: 0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0}
    G.add_edges_from([(0, 0), (1, 1), (2, 2)])
    assert nx.square_clustering(G) == {0: 1, 1: 0.5, 2: 0.2, 3: 0.0, 4: 0}
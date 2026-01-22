import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.bipartite.cluster import cc_dot, cc_max, cc_min
def test_average_path_graph():
    G = nx.path_graph(4)
    assert bipartite.average_clustering(G, mode='dot') == 0.5
    assert bipartite.average_clustering(G, mode='max') == 0.5
    assert bipartite.average_clustering(G, mode='min') == 1
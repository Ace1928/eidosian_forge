import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_selfloop_edges_multi_with_data_and_keys():
    G = nx.complete_graph(3, create_using=nx.MultiGraph)
    G.add_edge(0, 0, weight=10)
    G.add_edge(0, 0, weight=100)
    assert edges_equal(nx.selfloop_edges(G, data='weight', keys=True), [(0, 0, 0, 10), (0, 0, 1, 100)])
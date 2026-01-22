import itertools
import pytest
import networkx as nx
def sl_hc():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
    graph.add_edges_from([(1, 2), (1, 3), (1, 5), (1, 7), (2, 3), (2, 4), (2, 8), (8, 4), (8, 6), (8, 7), (7, 5), (7, 6), (3, 4), (4, 6), (6, 5), (5, 3)])
    return graph
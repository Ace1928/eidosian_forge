import itertools
import pytest
import networkx as nx
def slf_hc():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
    graph.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8)])
    return graph
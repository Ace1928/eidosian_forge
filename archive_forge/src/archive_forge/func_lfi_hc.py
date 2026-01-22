import itertools
import pytest
import networkx as nx
def lfi_hc():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9])
    graph.add_edges_from([(1, 2), (1, 5), (1, 6), (1, 7), (2, 3), (2, 8), (2, 9), (3, 4), (3, 8), (3, 9), (4, 5), (4, 6), (4, 7), (5, 6)])
    return graph
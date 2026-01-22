import itertools
import pytest
import networkx as nx
def sli_shc():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7])
    graph.add_edges_from([(1, 2), (1, 3), (1, 5), (1, 7), (2, 3), (2, 6), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7)])
    return graph
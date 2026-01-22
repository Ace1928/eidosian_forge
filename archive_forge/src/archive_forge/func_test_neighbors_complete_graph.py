import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_neighbors_complete_graph(self):
    graph = nx.complete_graph(100)
    pop = random.sample(list(graph), 1)
    nbors = list(nx.neighbors(graph, pop[0]))
    assert len(nbors) == len(graph) - 1
    graph = nx.path_graph(100)
    node = random.sample(list(graph), 1)[0]
    nbors = list(nx.neighbors(graph, node))
    if node != 0 and node != 99:
        assert len(nbors) == 2
    else:
        assert len(nbors) == 1
    graph = nx.star_graph(99)
    nbors = list(nx.neighbors(graph, 0))
    assert len(nbors) == 99
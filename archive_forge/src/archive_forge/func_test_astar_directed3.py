import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_directed3(self):
    heuristic_values = {'n5': 36, 'n2': 4, 'n1': 0, 'n0': 0}

    def h(u, v):
        return heuristic_values[u]
    edges = [('n5', 'n1', 11), ('n5', 'n2', 9), ('n2', 'n1', 1), ('n1', 'n0', 32)]
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges)
    answer = ['n5', 'n2', 'n1', 'n0']
    assert nx.astar_path(graph, 'n5', 'n0', h) == answer
import pytest
import networkx as nx
from networkx.utils import pairwise
def test_multiple_optimal_paths(self):
    """Tests that A* algorithm finds any of multiple optimal paths"""
    heuristic_values = {'a': 1.35, 'b': 1.18, 'c': 0.67, 'd': 0}

    def h(u, v):
        return heuristic_values[u]
    graph = nx.Graph()
    points = ['a', 'b', 'c', 'd']
    edges = [('a', 'b', 0.18), ('a', 'c', 0.68), ('b', 'c', 0.5), ('c', 'd', 0.67)]
    graph.add_nodes_from(points)
    graph.add_weighted_edges_from(edges)
    path1 = ['a', 'c', 'd']
    path2 = ['a', 'b', 'c', 'd']
    assert nx.astar_path(graph, 'a', 'd', h) in (path1, path2)
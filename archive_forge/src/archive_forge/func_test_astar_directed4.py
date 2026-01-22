import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_directed4(self):
    edges = [('a', 'b', 1), ('a', 'c', 1), ('b', 'd', 2), ('c', 'd', 1), ('d', 'e', 1)]
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges)
    assert nx.astar_path(graph, 'a', 'e') == ['a', 'c', 'd', 'e']
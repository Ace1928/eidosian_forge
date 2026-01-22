import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_mst():
    G = nx.Graph()
    edgelist = [(0, 3, [('weight', 5)]), (0, 1, [('weight', 7)]), (1, 3, [('weight', 9)]), (1, 2, [('weight', 8)]), (1, 4, [('weight', 7)]), (3, 4, [('weight', 15)]), (3, 5, [('weight', 6)]), (2, 4, [('weight', 5)]), (4, 5, [('weight', 8)]), (4, 6, [('weight', 9)]), (5, 6, [('weight', 11)])]
    G.add_edges_from(edgelist)
    G = G.to_directed()
    x = branchings.minimum_spanning_arborescence(G)
    edges = [({0, 1}, 7), ({0, 3}, 5), ({3, 5}, 6), ({1, 4}, 7), ({4, 2}, 5), ({4, 6}, 9)]
    assert x.number_of_edges() == len(edges)
    for u, v, d in x.edges(data=True):
        assert ({u, v}, d['weight']) in edges
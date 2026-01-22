import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_vertex_cover_issue_2384(self):
    G = nx.Graph([(0, 3), (1, 3), (1, 4), (2, 3)])
    matching = maximum_matching(G)
    vertex_cover = to_vertex_cover(G, matching)
    for u, v in G.edges():
        assert u in vertex_cover or v in vertex_cover
import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_vertex_cover_issue_3306(self):
    G = nx.Graph()
    edges = [(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)]
    G.add_edges_from([((i, 'L'), (j, 'R')) for i, j in edges])
    matching = maximum_matching(G)
    vertex_cover = to_vertex_cover(G, matching)
    for u, v in G.edges():
        assert u in vertex_cover or v in vertex_cover
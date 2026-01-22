import pytest
import networkx as nx
from networkx.utils import pairwise
def test_two_sources(self):
    edges = [(0, 1, 1), (1, 2, 1), (2, 3, 10), (3, 4, 1)]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    sources = {0, 4}
    distances, paths = nx.multi_source_dijkstra(G, sources)
    expected_distances = {0: 0, 1: 1, 2: 2, 3: 1, 4: 0}
    expected_paths = {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [4, 3], 4: [4]}
    assert distances == expected_distances
    assert paths == expected_paths
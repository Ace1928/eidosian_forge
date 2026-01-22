import networkx as nx
from networkx.utils import pairwise
def test_undirected_weighted(self):
    edges = [(0, 1, 10), (1, 2, 1), (2, 3, 1)]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    cells = nx.voronoi_cells(G, {0, 3})
    expected = {0: {0}, 3: {1, 2, 3}}
    assert expected == cells
import networkx as nx
from networkx.utils import pairwise
def test_multidigraph_unweighted(self):
    edges = list(pairwise(range(6), cyclic=True))
    G = nx.MultiDiGraph(2 * edges)
    H = nx.DiGraph(G)
    G_cells = nx.voronoi_cells(G, {0, 3})
    H_cells = nx.voronoi_cells(H, {0, 3})
    assert G_cells == H_cells
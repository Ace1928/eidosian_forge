from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_lattice_points(self):
    """Tests that the graph is really a hexagonal lattice."""
    for m, n in [(4, 5), (4, 4), (4, 3), (3, 2), (3, 3), (3, 5)]:
        G = nx.hexagonal_lattice_graph(m, n)
        assert len(G) == 2 * (m + 1) * (n + 1) - 2
    C_6 = nx.cycle_graph(6)
    hexagons = [[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)], [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)], [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)], [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4)]]
    for hexagon in hexagons:
        assert nx.is_isomorphic(G.subgraph(hexagon), C_6)
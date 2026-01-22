from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_periodic_iterable(self):
    m, n, k = (3, 7, 5)
    for a, b, c in product([0, 1], [0, 1], [0, 1]):
        G = nx.grid_graph([m, n, k], periodic=(a, b, c))
        num_e = (m + a - 1) * n * k + (n + b - 1) * m * k + (k + c - 1) * m * n
        assert G.number_of_nodes() == m * n * k
        assert G.number_of_edges() == num_e
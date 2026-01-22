from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_kemeny_constant_path_graph(self):
    n = 10
    G = nx.path_graph(n)
    K = nx.kemeny_constant(G)
    assert np.isclose(K, n ** 2 / 3 - 2 * n / 3 + 1 / 2)
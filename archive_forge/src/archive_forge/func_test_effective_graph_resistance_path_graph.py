from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_effective_graph_resistance_path_graph(self):
    N = 10
    G = nx.path_graph(N)
    RG = nx.effective_graph_resistance(G)
    assert np.isclose(RG, (N - 1) * N * (N + 1) // 6)
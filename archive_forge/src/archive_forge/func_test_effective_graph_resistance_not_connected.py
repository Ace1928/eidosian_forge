from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_effective_graph_resistance_not_connected(self):
    G = nx.Graph([(1, 2), (3, 4)])
    RG = nx.effective_graph_resistance(G)
    assert np.isinf(RG)
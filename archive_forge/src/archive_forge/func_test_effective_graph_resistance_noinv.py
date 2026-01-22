from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_effective_graph_resistance_noinv(self):
    RG = nx.effective_graph_resistance(self.G, 'weight', False)
    rd12 = 1 / (1 / (1 / 1 + 1 / 4) + 1 / (1 / 2))
    rd13 = 1 / (1 / (1 / 1 + 1 / 2) + 1 / (1 / 4))
    rd23 = 1 / (1 / (1 / 2 + 1 / 4) + 1 / (1 / 1))
    assert np.isclose(RG, rd12 + rd13 + rd23)
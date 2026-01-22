from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_resistance_distance_noinv(self):
    rd = nx.resistance_distance(self.G, 1, 3, 'weight', False)
    test_data = 1 / (1 / (1 / 2 + 1 / 4) + 1 / (1 / 1 + 1 / 3))
    assert round(rd, 5) == round(test_data, 5)
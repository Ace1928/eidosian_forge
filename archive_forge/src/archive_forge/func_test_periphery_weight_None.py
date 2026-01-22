from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_periphery_weight_None(self):
    for v in set(nx.periphery(self.G, weight=None)):
        assert nx.eccentricity(self.G, v, weight=None) == nx.diameter(self.G, weight=None)
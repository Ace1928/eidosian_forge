from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_eccentricity_weight_fn(self):
    assert nx.eccentricity(self.G, 1, weight=self.weight_fn) == 6
    e = nx.eccentricity(self.G, weight=self.weight_fn)
    assert e[1] == 6
    e = nx.eccentricity(self.G, v=1, weight=self.weight_fn)
    assert e == 6
    e = nx.eccentricity(self.G, v=[1, 1], weight=self.weight_fn)
    assert e[1] == 6
    e = nx.eccentricity(self.G, v=[1, 2], weight=self.weight_fn)
    assert e[1] == 6
from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_resistance_distance_div0(self):
    with pytest.raises(ZeroDivisionError):
        self.G[1][2]['weight'] = 0
        nx.resistance_distance(self.G, 1, 3, 'weight')
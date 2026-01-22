from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_resistance_distance_nodeA_not_in_graph(self):
    with pytest.raises(nx.NetworkXError):
        nx.resistance_distance(self.G, 9, 1)
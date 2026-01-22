from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_bound_diameter_weight_attr(self):
    assert nx.diameter(self.G, usebounds=True, weight='high_cost') != nx.diameter(self.G, usebounds=True, weight='weight') == nx.diameter(self.G, usebounds=True, weight='cost') == 1.6 != nx.diameter(self.G, usebounds=True, weight='high_cost')
    assert nx.diameter(self.G, usebounds=True, weight='high_cost') == nx.diameter(self.G, usebounds=True, weight='high_cost')
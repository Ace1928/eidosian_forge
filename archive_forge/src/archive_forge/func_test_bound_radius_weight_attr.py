from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_bound_radius_weight_attr(self):
    assert nx.radius(self.G, usebounds=True, weight='high_cost') != pytest.approx(nx.radius(self.G, usebounds=True, weight='weight')) == pytest.approx(nx.radius(self.G, usebounds=True, weight='cost')) == 0.9 != nx.radius(self.G, usebounds=True, weight='high_cost')
    assert nx.radius(self.G, usebounds=True, weight='high_cost') == nx.radius(self.G, usebounds=True, weight='high_cost')
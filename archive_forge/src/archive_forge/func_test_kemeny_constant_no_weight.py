from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_kemeny_constant_no_weight(self):
    K = nx.kemeny_constant(self.G)
    assert np.isclose(K, 4 / 3)
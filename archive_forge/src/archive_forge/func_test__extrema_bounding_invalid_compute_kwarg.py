from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test__extrema_bounding_invalid_compute_kwarg():
    G = nx.path_graph(3)
    with pytest.raises(ValueError, match='compute must be one of'):
        _extrema_bounding(G, compute='spam')
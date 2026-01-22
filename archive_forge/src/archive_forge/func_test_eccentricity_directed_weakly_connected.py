from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_eccentricity_directed_weakly_connected(self):
    with pytest.raises(nx.NetworkXError):
        DG = nx.DiGraph([(1, 2), (1, 3)])
        nx.eccentricity(DG)
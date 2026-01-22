from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_eccentricity_undirected_not_connected(self):
    with pytest.raises(nx.NetworkXError):
        G = nx.Graph([(1, 2), (3, 4)])
        e = nx.eccentricity(G, sp=1)
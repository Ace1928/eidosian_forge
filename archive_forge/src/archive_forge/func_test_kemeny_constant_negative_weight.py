from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_kemeny_constant_negative_weight(self):
    G = nx.Graph()
    w12 = 2
    w13 = 3
    w23 = -10
    G.add_edge(1, 2, weight=w12)
    G.add_edge(1, 3, weight=w13)
    G.add_edge(2, 3, weight=w23)
    with pytest.raises(nx.NetworkXError):
        nx.kemeny_constant(G, weight='weight')
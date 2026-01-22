from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_effective_graph_resistance_div0(self):
    with pytest.raises(ZeroDivisionError):
        self.G[1][2]['weight'] = 0
        nx.effective_graph_resistance(self.G, 'weight')
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_degree2(self):
    H = self.G()
    H.add_edges_from([(1, 24), (1, 2)])
    assert sorted((d for n, d in H.degree([1, 24]))) == [1, 2]
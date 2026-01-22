import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_number_of_edges(self):
    assert self.G.number_of_edges() == nx.number_of_edges(self.G)
    assert self.DG.number_of_edges() == nx.number_of_edges(self.DG)
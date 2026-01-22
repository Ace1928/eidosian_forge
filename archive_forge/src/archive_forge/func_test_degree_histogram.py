import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_degree_histogram(self):
    assert nx.degree_histogram(self.G) == [1, 1, 1, 1, 1]
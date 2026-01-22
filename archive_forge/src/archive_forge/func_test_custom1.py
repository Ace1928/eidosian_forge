import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_custom1(self):
    """Case of no common neighbors."""
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    self.test(G, 0, 1, [])
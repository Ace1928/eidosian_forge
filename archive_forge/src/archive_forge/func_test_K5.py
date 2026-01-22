import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_K5(self):
    G = nx.complete_graph(5)
    self.test(G, 0, 1, [2, 3, 4])
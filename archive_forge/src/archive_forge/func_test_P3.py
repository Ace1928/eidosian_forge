import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_P3(self):
    G = nx.path_graph(3)
    self.test(G, 0, 2, [1])
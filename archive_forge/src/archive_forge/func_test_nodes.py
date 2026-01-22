import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nodes(self):
    assert nodes_equal(self.G.nodes(), list(nx.nodes(self.G)))
    assert nodes_equal(self.DG.nodes(), list(nx.nodes(self.DG)))
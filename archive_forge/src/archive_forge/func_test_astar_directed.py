import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_directed(self):
    assert nx.astar_path(self.XG, 's', 'v') == ['s', 'x', 'u', 'v']
    assert nx.astar_path_length(self.XG, 's', 'v') == 9
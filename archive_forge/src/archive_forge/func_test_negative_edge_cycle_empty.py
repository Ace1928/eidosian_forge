import pytest
import networkx as nx
from networkx.utils import pairwise
def test_negative_edge_cycle_empty(self):
    G = nx.DiGraph()
    assert not nx.negative_edge_cycle(G)
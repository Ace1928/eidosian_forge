import os
import tempfile
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_parse_pajek(self):
    G = nx.parse_pajek(self.data)
    assert sorted(G.nodes()) == ['A1', 'Bb', 'C', 'D2']
    assert edges_equal(G.edges(), [('A1', 'A1'), ('A1', 'Bb'), ('A1', 'C'), ('Bb', 'A1'), ('C', 'C'), ('C', 'D2'), ('D2', 'Bb')])
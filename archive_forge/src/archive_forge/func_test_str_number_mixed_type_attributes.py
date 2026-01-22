import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_str_number_mixed_type_attributes(self):
    G = nx.MultiGraph()
    G.add_node('n0', special='hello')
    G.add_node('n1', special=0)
    G.add_edge('n0', 'n1', special='hello')
    G.add_edge('n0', 'n1', special=0)
    fh = io.BytesIO()
    self.writer(G, fh)
    fh.seek(0)
    H = nx.read_graphml(fh)
    assert H.nodes['n0']['special'] == 'hello'
    assert H.nodes['n1']['special'] == 0
    assert H.edges['n0', 'n1', 0]['special'] == 'hello'
    assert H.edges['n0', 'n1', 1]['special'] == 0
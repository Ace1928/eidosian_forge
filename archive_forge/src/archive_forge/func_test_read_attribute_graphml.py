import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_read_attribute_graphml(self):
    G = self.attribute_graph
    H = nx.read_graphml(self.attribute_fh)
    assert nodes_equal(G.nodes(True), sorted(H.nodes(data=True)))
    ge = sorted(G.edges(data=True))
    he = sorted(H.edges(data=True))
    for a, b in zip(ge, he):
        assert a == b
    self.attribute_fh.seek(0)
    PG = nx.parse_graphml(self.attribute_data)
    assert sorted(G.nodes(True)) == sorted(PG.nodes(data=True))
    ge = sorted(G.edges(data=True))
    he = sorted(PG.edges(data=True))
    for a, b in zip(ge, he):
        assert a == b
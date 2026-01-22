import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_mixed_int_type_number_attributes(self):
    np = pytest.importorskip('numpy')
    G = nx.MultiGraph()
    G.add_node('n0', special=np.int64(0))
    G.add_node('n1', special=1)
    G.add_edge('n0', 'n1', special=np.int64(2))
    G.add_edge('n0', 'n1', special=3)
    fh = io.BytesIO()
    self.writer(G, fh)
    fh.seek(0)
    H = nx.read_graphml(fh)
    assert H.nodes['n0']['special'] == 0
    assert H.nodes['n1']['special'] == 1
    assert H.edges['n0', 'n1', 0]['special'] == 2
    assert H.edges['n0', 'n1', 1]['special'] == 3
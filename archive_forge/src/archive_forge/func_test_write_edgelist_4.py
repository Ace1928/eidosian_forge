import io
import os
import tempfile
import textwrap
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_write_edgelist_4(self):
    fh = io.BytesIO()
    G = nx.Graph()
    G.add_edge(1, 2, weight=2.0)
    G.add_edge(2, 3, weight=3.0)
    nx.write_edgelist(G, fh, data=['weight'])
    fh.seek(0)
    assert fh.read() == b'1 2 2.0\n2 3 3.0\n'
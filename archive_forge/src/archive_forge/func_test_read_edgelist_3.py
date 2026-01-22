import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_read_edgelist_3(self):
    s = b"# comment line\n1 2 {'weight':2.0}\n# comment line\n2 3 {'weight':3.0}\n"
    bytesIO = io.BytesIO(s)
    G = bipartite.read_edgelist(bytesIO, nodetype=int, data=False)
    assert edges_equal(G.edges(), [(1, 2), (2, 3)])
    bytesIO = io.BytesIO(s)
    G = bipartite.read_edgelist(bytesIO, nodetype=int, data=True)
    assert edges_equal(G.edges(data=True), [(1, 2, {'weight': 2.0}), (2, 3, {'weight': 3.0})])
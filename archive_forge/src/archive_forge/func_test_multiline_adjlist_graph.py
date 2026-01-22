import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_multiline_adjlist_graph(self):
    G = self.G
    fd, fname = tempfile.mkstemp()
    nx.write_multiline_adjlist(G, fname)
    H = nx.read_multiline_adjlist(fname)
    H2 = nx.read_multiline_adjlist(fname)
    assert H is not H2
    assert nodes_equal(list(H), list(G))
    assert edges_equal(list(H.edges()), list(G.edges()))
    os.close(fd)
    os.unlink(fname)
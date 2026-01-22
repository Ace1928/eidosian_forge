import io
import os
import tempfile
import textwrap
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_edgelist_graph(self):
    G = self.G
    fd, fname = tempfile.mkstemp()
    nx.write_edgelist(G, fname)
    H = nx.read_edgelist(fname)
    H2 = nx.read_edgelist(fname)
    assert H is not H2
    G.remove_node('g')
    assert nodes_equal(list(H), list(G))
    assert edges_equal(list(H.edges()), list(G.edges()))
    os.close(fd)
    os.unlink(fname)
import io
import os
import tempfile
import textwrap
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_edgelist_multidigraph(self):
    G = self.XDG
    fd, fname = tempfile.mkstemp()
    nx.write_edgelist(G, fname)
    H = nx.read_edgelist(fname, nodetype=int, create_using=nx.MultiDiGraph())
    H2 = nx.read_edgelist(fname, nodetype=int, create_using=nx.MultiDiGraph())
    assert H is not H2
    assert nodes_equal(list(H), list(G))
    assert edges_equal(list(H.edges()), list(G.edges()))
    os.close(fd)
    os.unlink(fname)
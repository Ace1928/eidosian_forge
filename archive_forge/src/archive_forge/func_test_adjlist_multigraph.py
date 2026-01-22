import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_adjlist_multigraph(self):
    G = self.XG
    fd, fname = tempfile.mkstemp()
    nx.write_adjlist(G, fname)
    H = nx.read_adjlist(fname, nodetype=int, create_using=nx.MultiGraph())
    H2 = nx.read_adjlist(fname, nodetype=int, create_using=nx.MultiGraph())
    assert H is not H2
    assert nodes_equal(list(H), list(G))
    assert edges_equal(list(H.edges()), list(G.edges()))
    os.close(fd)
    os.unlink(fname)
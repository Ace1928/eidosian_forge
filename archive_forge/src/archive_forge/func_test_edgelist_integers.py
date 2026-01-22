import io
import os
import tempfile
import textwrap
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_edgelist_integers(self):
    G = nx.convert_node_labels_to_integers(self.G)
    fd, fname = tempfile.mkstemp()
    nx.write_edgelist(G, fname)
    H = nx.read_edgelist(fname, nodetype=int)
    G.remove_nodes_from(list(nx.isolates(G)))
    assert nodes_equal(list(H), list(G))
    assert edges_equal(list(H.edges()), list(G.edges()))
    os.close(fd)
    os.unlink(fname)
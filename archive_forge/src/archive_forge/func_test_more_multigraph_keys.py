import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_more_multigraph_keys(self):
    """Writing keys as edge id attributes means keys become strings.
        The original keys are stored as data, so read them back in
        if `str(key) == edge_id`
        This allows the adjacency to remain the same.
        """
    G = nx.MultiGraph()
    G.add_edges_from([('a', 'b', 2), ('a', 'b', 3)])
    fd, fname = tempfile.mkstemp()
    self.writer(G, fname)
    H = nx.read_graphml(fname)
    assert H.is_multigraph()
    assert edges_equal(G.edges(keys=True), H.edges(keys=True))
    assert G._adj == H._adj
    os.close(fd)
    os.unlink(fname)
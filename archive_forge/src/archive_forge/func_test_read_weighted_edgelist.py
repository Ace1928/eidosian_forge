import io
import os
import tempfile
import textwrap
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_read_weighted_edgelist():
    bytesIO = io.BytesIO(edges_with_values.encode('utf-8'))
    G = nx.read_weighted_edgelist(bytesIO, nodetype=int)
    assert edges_equal(G.edges(data=True), _expected_edges_weights)
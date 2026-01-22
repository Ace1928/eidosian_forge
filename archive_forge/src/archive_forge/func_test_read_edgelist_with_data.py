import io
import os
import tempfile
import textwrap
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
@pytest.mark.parametrize(('data', 'extra_kwargs', 'expected'), ((edges_with_weight, {}, _expected_edges_weights), (edges_with_multiple_attrs, {}, _expected_edges_multiattr), (edges_with_multiple_attrs_csv, {'delimiter': ','}, _expected_edges_multiattr)))
def test_read_edgelist_with_data(data, extra_kwargs, expected):
    bytesIO = io.BytesIO(data.encode('utf-8'))
    G = nx.read_edgelist(bytesIO, nodetype=int, **extra_kwargs)
    assert edges_equal(G.edges(data=True), expected)
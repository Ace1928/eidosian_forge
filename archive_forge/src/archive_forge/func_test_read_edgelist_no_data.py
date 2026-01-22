import io
import os
import tempfile
import textwrap
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
@pytest.mark.parametrize(('data', 'extra_kwargs'), ((edges_no_data, {}), (edges_with_values, {}), (edges_with_weight, {}), (edges_with_multiple_attrs, {}), (edges_with_multiple_attrs_csv, {'delimiter': ','})))
def test_read_edgelist_no_data(data, extra_kwargs):
    bytesIO = io.BytesIO(data.encode('utf-8'))
    G = nx.read_edgelist(bytesIO, nodetype=int, data=False, **extra_kwargs)
    assert edges_equal(G.edges(), [(1, 2), (2, 3)])
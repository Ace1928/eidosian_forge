import os
import tempfile
from io import StringIO
import pytest
import networkx as nx
from networkx.utils import graphs_equal
def test_pydot_numerical_name():
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), (0, 1)])
    graph_layout = nx.nx_pydot.pydot_layout(G, prog='dot')
    assert isinstance(graph_layout, dict)
    assert '0' not in graph_layout
    assert 0 in graph_layout
    assert '1' not in graph_layout
    assert 1 in graph_layout
    assert 'A' in graph_layout
    assert 'B' in graph_layout
import io
import os
import tempfile
import textwrap
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_parse_edgelist():
    lines = ['1;2', '2 3', '3 4']
    G = nx.parse_edgelist(lines, nodetype=int)
    assert list(G.edges()) == [(2, 3), (3, 4)]
    with pytest.raises(TypeError, match='Failed to convert nodes'):
        lines = ['1 2', '2 3', '3 4']
        nx.parse_edgelist(lines, nodetype='nope')
    with pytest.raises(TypeError, match='Failed to convert edge data'):
        lines = ['1 2 3', '2 3', '3 4']
        nx.parse_edgelist(lines, nodetype=int)
    with pytest.raises(IndexError, match='not the same length'):
        lines = ['1 2 3', '2 3 27', '3 4 3.0']
        nx.parse_edgelist(lines, nodetype=int, data=(('weight', float), ('capacity', int)))
    with pytest.raises(TypeError, match='Failed to convert'):
        lines = ["1 2 't1'", "2 3 't3'", "3 4 't3'"]
        nx.parse_edgelist(lines, nodetype=int, data=(('weight', float),))
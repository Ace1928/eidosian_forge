import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_parse_multiline_adjlist(self):
    lines = ['1 2', "b {'weight':3, 'name': 'Frodo'}", 'c {}', 'd 1', "e {'weight':6, 'name': 'Saruman'}"]
    nx.parse_multiline_adjlist(iter(lines))
    with pytest.raises(TypeError):
        nx.parse_multiline_adjlist(iter(lines), nodetype=int)
    nx.parse_multiline_adjlist(iter(lines), edgetype=str)
    with pytest.raises(TypeError):
        nx.parse_multiline_adjlist(iter(lines), nodetype=int)
    lines = ['1 a']
    with pytest.raises(TypeError):
        nx.parse_multiline_adjlist(iter(lines))
    lines = ['a 2']
    with pytest.raises(TypeError):
        nx.parse_multiline_adjlist(iter(lines), nodetype=int)
    lines = ['1 2']
    with pytest.raises(TypeError):
        nx.parse_multiline_adjlist(iter(lines))
    lines = ['1 2', '2 {}']
    with pytest.raises(TypeError):
        nx.parse_multiline_adjlist(iter(lines))
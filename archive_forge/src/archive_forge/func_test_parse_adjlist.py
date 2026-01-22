import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_parse_adjlist(self):
    lines = ['1 2 5', '2 3 4', '3 5', '4', '5']
    nx.parse_adjlist(lines, nodetype=int)
    with pytest.raises(TypeError):
        nx.parse_adjlist(lines, nodetype='int')
    lines = ['1 2 5', '2 b', 'c']
    with pytest.raises(TypeError):
        nx.parse_adjlist(lines, nodetype=int)
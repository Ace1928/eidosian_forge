import io
import os
import tempfile
import textwrap
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_comments_None():
    edgelist = ['node#1 node#2', 'node#2 node#3']
    G = nx.parse_edgelist(edgelist, comments=None)
    H = nx.Graph([e.split(' ') for e in edgelist])
    assert edges_equal(G.edges, H.edges)
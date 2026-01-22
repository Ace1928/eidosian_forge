import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_raise_attribute(self):
    with pytest.raises(AttributeError):
        G = nx.path_graph(4)
        bytesIO = io.BytesIO()
        bipartite.write_edgelist(G, bytesIO)
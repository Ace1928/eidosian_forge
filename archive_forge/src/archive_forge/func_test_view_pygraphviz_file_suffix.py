import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_view_pygraphviz_file_suffix(self, tmp_path):
    G = nx.complete_graph(3)
    path, A = nx.nx_agraph.view_pygraphviz(G, suffix=1, show=False)
    assert path[-6:] == '_1.png'
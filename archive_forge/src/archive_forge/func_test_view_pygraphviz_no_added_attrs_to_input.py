import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_view_pygraphviz_no_added_attrs_to_input(self):
    G = nx.complete_graph(2)
    path, A = nx.nx_agraph.view_pygraphviz(G, show=False)
    assert G.graph == {}
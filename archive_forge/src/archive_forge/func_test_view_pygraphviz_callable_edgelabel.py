import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_view_pygraphviz_callable_edgelabel(self):
    G = nx.complete_graph(3)

    def foo_label(data):
        return 'foo'
    path, A = nx.nx_agraph.view_pygraphviz(G, edgelabel=foo_label, show=False)
    for edge in A.edges():
        assert edge.attr['label'] == 'foo'
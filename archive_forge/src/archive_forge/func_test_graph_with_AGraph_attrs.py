import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_graph_with_AGraph_attrs(self):
    G = nx.complete_graph(2)
    G.graph['node'] = {'width': '0.80'}
    G.graph['edge'] = {'fontsize': '14'}
    path, A = nx.nx_agraph.view_pygraphviz(G, show=False)
    assert dict(A.node_attr)['width'] == '0.80'
    assert dict(A.edge_attr)['fontsize'] == '14'
import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
@pytest.mark.xfail(reason='known bug in clean_attrs')
def test_view_pygraphviz_leaves_input_graph_unmodified(self):
    G = nx.complete_graph(2)
    G.graph['node'] = {'width': '0.80'}
    G.graph['edge'] = {'fontsize': '14'}
    path, A = nx.nx_agraph.view_pygraphviz(G, show=False)
    assert G.graph == {'node': {'width': '0.80'}, 'edge': {'fontsize': '14'}}
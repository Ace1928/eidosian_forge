import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
@pytest.mark.parametrize('root', range(5))
def test_pygraphviz_layout_root(self, root):
    G = nx.complete_graph(5)
    A = nx.nx_agraph.to_agraph(G)
    pygv_layout = nx.nx_agraph.pygraphviz_layout(G, prog='circo', root=root)
    A.layout(args=f'-Groot={root}', prog='circo')
    a1_pos = tuple((float(v) for v in dict(A.get_node('1').attr)['pos'].split(',')))
    assert pygv_layout[1] == a1_pos
import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_3d_layout(self):
    G = nx.Graph()
    G = self.build_graph(G)
    G.graph['dimen'] = 3
    pos = nx.nx_agraph.pygraphviz_layout(G, prog='neato')
    pos = list(pos.values())
    assert len(pos) == 5
    assert len(pos[0]) == 3
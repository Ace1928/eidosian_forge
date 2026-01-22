import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_from_agraph_named_edges(self):
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    A = nx.nx_agraph.to_agraph(G)
    A.add_edge(0, 1, key='foo')
    H = nx.nx_agraph.from_agraph(A)
    assert isinstance(H, nx.Graph)
    assert ('0', '1', {'key': 'foo'}) in H.edges(data=True)
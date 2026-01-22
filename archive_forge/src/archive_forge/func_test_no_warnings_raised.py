import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_no_warnings_raised(self):
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(1, 1))
    A = nx.nx_agraph.to_agraph(G)
    with pytest.warns(None) as record:
        A.layout()
    assert len(record) == 0
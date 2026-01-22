import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
@pytest.mark.xfail(reason='integer->string node conversion in round trip')
def test_round_trip_integer_nodes(self):
    G = nx.complete_graph(3)
    A = nx.nx_agraph.to_agraph(G)
    H = nx.nx_agraph.from_agraph(A)
    assert graphs_equal(G, H)
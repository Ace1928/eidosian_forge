import os
import tempfile
from io import StringIO
import pytest
import networkx as nx
from networkx.utils import graphs_equal
@pytest.mark.parametrize('graph_type', [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_hashable_pydot(graph_type):
    G = graph_type()
    G.add_edge('5', frozenset([1]), t='"Example:A"', l=False)
    G.add_edge('1', 2, w=True, t=('node1',), l=frozenset(['node1']))
    G.add_edge('node', (3, 3), w='string')
    assert [{'t': '"Example:A"', 'l': 'False'}, {'w': 'True', 't': "('node1',)", 'l': "frozenset({'node1'})"}, {'w': 'string'}] == [attr for _, _, attr in nx.nx_pydot.from_pydot(nx.nx_pydot.to_pydot(G)).edges.data()]
    assert {str(i) for i in G.nodes()} == set(nx.nx_pydot.from_pydot(nx.nx_pydot.to_pydot(G)).nodes)
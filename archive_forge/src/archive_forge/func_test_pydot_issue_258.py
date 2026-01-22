import os
import tempfile
from io import StringIO
import pytest
import networkx as nx
from networkx.utils import graphs_equal
def test_pydot_issue_258():
    G = nx.Graph([('Example:A', 1)])
    with pytest.raises(ValueError):
        nx.nx_pydot.to_pydot(G)
    with pytest.raises(ValueError):
        nx.nx_pydot.pydot_layout(G)
    G = nx.Graph()
    G.add_node('1.2', style='filled', fillcolor='red:yellow')
    with pytest.raises(ValueError):
        nx.nx_pydot.to_pydot(G)
    G.remove_node('1.2')
    G.add_node('1.2', style='filled', fillcolor='"red:yellow"')
    assert G.nodes.data() == nx.nx_pydot.from_pydot(nx.nx_pydot.to_pydot(G)).nodes.data()
    G = nx.DiGraph()
    G.add_edge('1', '2', foo='bar:1')
    with pytest.raises(ValueError):
        nx.nx_pydot.to_pydot(G)
    G = nx.DiGraph()
    G.add_edge('1', '2', foo='"bar:1"')
    assert G['1']['2'] == nx.nx_pydot.from_pydot(nx.nx_pydot.to_pydot(G))['1']['2']
    G = nx.MultiGraph()
    G.add_edge('1', '2', foo='b:1')
    G.add_edge('1', '2', bar='foo:foo')
    with pytest.raises(ValueError):
        nx.nx_pydot.to_pydot(G)
    G = nx.MultiGraph()
    G.add_edge('1', '2', foo='"b:1"')
    G.add_edge('1', '2', bar='"foo:foo"')
    assert [attr for _, _, attr in G.edges.data()] == [attr for _, _, attr in nx.nx_pydot.from_pydot(nx.nx_pydot.to_pydot(G)).edges.data()]
    G = nx.Graph()
    G.add_edge('1', '2')
    G['1']['2']['f:oo'] = 'bar'
    with pytest.raises(ValueError):
        nx.nx_pydot.to_pydot(G)
    G = nx.Graph()
    G.add_edge('1', '2')
    G['1']['2']['"f:oo"'] = 'bar'
    assert G['1']['2'] == nx.nx_pydot.from_pydot(nx.nx_pydot.to_pydot(G))['1']['2']
    G = nx.Graph([('"Example:A"', 1)])
    layout = nx.nx_pydot.pydot_layout(G)
    assert isinstance(layout, dict)
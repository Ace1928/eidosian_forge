import json
import pytest
import networkx as nx
from networkx.readwrite.json_graph import node_link_data, node_link_graph
def test_custom_attrs(self):
    G = nx.path_graph(4)
    G.add_node(1, color='red')
    G.add_edge(1, 2, width=7)
    G.graph[1] = 'one'
    G.graph['foo'] = 'bar'
    attrs = {'source': 'c_source', 'target': 'c_target', 'name': 'c_id', 'key': 'c_key', 'link': 'c_links'}
    H = node_link_graph(node_link_data(G, **attrs), multigraph=False, **attrs)
    assert nx.is_isomorphic(G, H)
    assert H.graph['foo'] == 'bar'
    assert H.nodes[1]['color'] == 'red'
    assert H[1][2]['width'] == 7
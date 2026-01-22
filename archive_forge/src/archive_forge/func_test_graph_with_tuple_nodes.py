import json
import pytest
import networkx as nx
from networkx.readwrite.json_graph import node_link_data, node_link_graph
def test_graph_with_tuple_nodes(self):
    G = nx.Graph()
    G.add_edge((0, 0), (1, 0), color=[255, 255, 0])
    d = node_link_data(G)
    dumped_d = json.dumps(d)
    dd = json.loads(dumped_d)
    H = node_link_graph(dd)
    assert H.nodes[0, 0] == G.nodes[0, 0]
    assert H[0, 0][1, 0]['color'] == [255, 255, 0]
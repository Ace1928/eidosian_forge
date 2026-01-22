import pytest
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_mcis(self):
    graph1 = nx.Graph()
    graph1.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (4, 5)])
    graph1.nodes[1]['color'] = 0
    graph2 = nx.Graph()
    graph2.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (3, 5), (5, 6), (5, 7), (6, 7)])
    graph2.nodes[1]['color'] = 1
    graph2.nodes[6]['color'] = 2
    graph2.nodes[7]['color'] = 2
    ismags = iso.ISMAGS(graph1, graph2, node_match=iso.categorical_node_match('color', None))
    assert list(ismags.subgraph_isomorphisms_iter(True)) == []
    assert list(ismags.subgraph_isomorphisms_iter(False)) == []
    found_mcis = _matches_to_sets(ismags.largest_common_subgraph())
    expected = _matches_to_sets([{2: 2, 3: 4, 4: 3, 5: 5}, {2: 4, 3: 2, 4: 3, 5: 5}])
    assert expected == found_mcis
    ismags = iso.ISMAGS(graph2, graph1, node_match=iso.categorical_node_match('color', None))
    assert list(ismags.subgraph_isomorphisms_iter(True)) == []
    assert list(ismags.subgraph_isomorphisms_iter(False)) == []
    found_mcis = _matches_to_sets(ismags.largest_common_subgraph())
    expected = _matches_to_sets([{2: 2, 3: 4, 4: 3, 5: 5}, {4: 2, 2: 3, 3: 4, 5: 5}])
    assert expected == found_mcis
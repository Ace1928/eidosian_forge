import pytest
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_symmetry_mcis(self):
    graph1 = nx.Graph()
    nx.add_path(graph1, range(4))
    graph2 = nx.Graph()
    nx.add_path(graph2, range(3))
    graph2.add_edge(1, 3)
    ismags1 = iso.ISMAGS(graph1, graph2, node_match=iso.categorical_node_match('color', None))
    assert list(ismags1.subgraph_isomorphisms_iter(True)) == []
    found_mcis = _matches_to_sets(ismags1.largest_common_subgraph())
    expected = _matches_to_sets([{0: 0, 1: 1, 2: 2}, {1: 0, 3: 2, 2: 1}])
    assert expected == found_mcis
    ismags2 = iso.ISMAGS(graph2, graph1, node_match=iso.categorical_node_match('color', None))
    assert list(ismags2.subgraph_isomorphisms_iter(True)) == []
    found_mcis = _matches_to_sets(ismags2.largest_common_subgraph())
    expected = _matches_to_sets([{3: 2, 0: 0, 1: 1}, {2: 0, 0: 2, 1: 1}, {3: 0, 0: 2, 1: 1}, {3: 0, 1: 1, 2: 2}, {0: 0, 1: 1, 2: 2}, {2: 0, 3: 2, 1: 1}])
    assert expected == found_mcis
    found_mcis1 = _matches_to_sets(ismags1.largest_common_subgraph(False))
    found_mcis2 = ismags2.largest_common_subgraph(False)
    found_mcis2 = [{v: k for k, v in d.items()} for d in found_mcis2]
    found_mcis2 = _matches_to_sets(found_mcis2)
    expected = _matches_to_sets([{3: 2, 1: 3, 2: 1}, {2: 0, 0: 2, 1: 1}, {1: 2, 3: 3, 2: 1}, {3: 0, 1: 3, 2: 1}, {0: 2, 2: 3, 1: 1}, {3: 0, 1: 2, 2: 1}, {2: 0, 0: 3, 1: 1}, {0: 0, 2: 3, 1: 1}, {1: 0, 3: 3, 2: 1}, {1: 0, 3: 2, 2: 1}, {0: 3, 1: 1, 2: 2}, {0: 0, 1: 1, 2: 2}])
    assert expected == found_mcis1
    assert expected == found_mcis2
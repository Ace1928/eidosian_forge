import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity.kcutsets import _is_separating_set
def test_all_node_cuts_simple_case():
    G = nx.complete_graph(5)
    G.remove_edges_from([(0, 1), (3, 4)])
    expected = [{0, 1, 2}, {2, 3, 4}]
    actual = list(nx.all_node_cuts(G))
    assert len(actual) == len(expected)
    for cut in actual:
        assert cut in expected
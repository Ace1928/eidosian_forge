import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_branchings_with_default_weights():
    """
    Tests that various brancing algorithms work on graphs without weights.
    For more information, see issue #7279.
    """
    graph = nx.erdos_renyi_graph(10, p=0.2, directed=True, seed=123)
    assert all(('weight' not in d for u, v, d in graph.edges(data=True))), 'test is for graphs without a weight attribute'
    nx.minimum_spanning_arborescence(graph.copy())
    nx.maximum_spanning_arborescence(graph.copy())
    nx.minimum_branching(graph.copy())
    nx.maximum_branching(graph.copy())
    nx.algorithms.tree.minimal_branching(graph.copy())
    nx.algorithms.tree.branching_weight(graph.copy())
    nx.algorithms.tree.greedy_branching(graph.copy())
    assert all(('weight' not in d for u, v, d in graph.edges(data=True))), 'The above calls should not modify the initial graph in-place'
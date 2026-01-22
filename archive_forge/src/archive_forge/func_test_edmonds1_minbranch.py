import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_edmonds1_minbranch():
    edges = [(u, v, -w) for u, v, w in optimal_arborescence_1]
    G = nx.from_numpy_array(-G_array, create_using=nx.DiGraph)
    x = branchings.maximum_branching(G)
    x_ = build_branching([])
    assert_equal_branchings(x, x_)
    x = branchings.minimum_branching(G)
    x_ = build_branching(edges)
    assert_equal_branchings(x, x_)
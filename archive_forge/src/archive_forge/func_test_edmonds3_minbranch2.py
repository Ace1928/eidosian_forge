import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_edmonds3_minbranch2():
    G = G1()
    G.add_edge(8, 9, weight=-10)
    x = branchings.minimum_branching(G)
    edges = [(8, 9, -10)]
    x_ = build_branching(edges)
    assert_equal_branchings(x, x_)
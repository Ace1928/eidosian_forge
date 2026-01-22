from itertools import permutations
import pytest
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
def test_partition_argument():
    G = nx.barbell_graph(3, 0)
    partition = [{0, 1, 2}, {3, 4, 5}]
    C = kernighan_lin_bisection(G, partition)
    assert_partition_equal(C, partition)
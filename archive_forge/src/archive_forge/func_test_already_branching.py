from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_already_branching(self):
    """Tests that a directed acyclic graph that is already a
        branching produces an isomorphic branching as output.

        """
    T1 = nx.balanced_tree(2, 2, create_using=nx.DiGraph())
    T2 = nx.balanced_tree(2, 2, create_using=nx.DiGraph())
    G = nx.disjoint_union(T1, T2)
    B = nx.dag_to_branching(G)
    assert nx.is_isomorphic(G, B)
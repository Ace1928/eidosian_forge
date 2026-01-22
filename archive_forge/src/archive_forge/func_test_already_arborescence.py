from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_already_arborescence(self):
    """Tests that a directed acyclic graph that is already an
        arborescence produces an isomorphic arborescence as output.

        """
    A = nx.balanced_tree(2, 2, create_using=nx.DiGraph())
    B = nx.dag_to_branching(A)
    assert nx.is_isomorphic(A, B)
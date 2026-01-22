import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_balanced_tree_path(self):
    """Tests that the balanced tree with branching factor one is the
        path graph.

        """
    T = nx.balanced_tree(1, 4)
    P = nx.path_graph(5)
    assert is_isomorphic(T, P)
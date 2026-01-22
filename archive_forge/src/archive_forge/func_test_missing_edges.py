from itertools import combinations
import pytest
from networkx import DiGraph
from networkx.algorithms.tournament import (
def test_missing_edges():
    """A tournament must not have any pair of nodes without at least
    one edge joining the pair.

    """
    G = DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])
    assert not is_tournament(G)
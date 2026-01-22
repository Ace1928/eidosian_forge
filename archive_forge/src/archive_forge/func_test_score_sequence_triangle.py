from itertools import combinations
import pytest
from networkx import DiGraph
from networkx.algorithms.tournament import (
def test_score_sequence_triangle():
    G = DiGraph([(0, 1), (1, 2), (2, 0)])
    assert score_sequence(G) == [1, 1, 1]
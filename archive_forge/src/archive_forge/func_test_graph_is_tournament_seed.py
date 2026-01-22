from itertools import combinations
import pytest
from networkx import DiGraph
from networkx.algorithms.tournament import (
def test_graph_is_tournament_seed():
    for _ in range(10):
        G = random_tournament(5, seed=1)
        assert is_tournament(G)
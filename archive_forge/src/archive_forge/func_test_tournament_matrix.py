from itertools import combinations
import pytest
from networkx import DiGraph
from networkx.algorithms.tournament import (
def test_tournament_matrix():
    np = pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    npt = np.testing
    G = DiGraph([(0, 1)])
    m = tournament_matrix(G)
    npt.assert_array_equal(m.todense(), np.array([[0, 1], [-1, 0]]))
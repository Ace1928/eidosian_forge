import pytest
import networkx as nx
def test_trophic_levels_simple():
    matrix_a = np.array([[0, 0], [1, 0]])
    G = nx.from_numpy_array(matrix_a, create_using=nx.DiGraph)
    d = nx.trophic_levels(G)
    assert d[0] == pytest.approx(2, abs=1e-07)
    assert d[1] == pytest.approx(1, abs=1e-07)
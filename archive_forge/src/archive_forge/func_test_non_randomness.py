import pytest
import networkx as nx
@pytest.mark.parametrize('k, weight, expected', [(None, None, 7.21), (2, None, 11.7), (None, 'weight', 25.45), (2, 'weight', 38.8)])
def test_non_randomness(k, weight, expected):
    G = nx.karate_club_graph()
    np.testing.assert_almost_equal(nx.non_randomness(G, k, weight)[0], expected, decimal=2)
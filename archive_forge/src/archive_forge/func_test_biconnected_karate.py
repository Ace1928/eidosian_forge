import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_biconnected_karate():
    K = nx.karate_club_graph()
    answer = [{0, 1, 2, 3, 7, 8, 9, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}, {0, 4, 5, 6, 10, 16}, {0, 11}]
    bcc = list(nx.biconnected_components(K))
    assert_components_equal(bcc, answer)
    assert set(nx.articulation_points(K)) == {0}
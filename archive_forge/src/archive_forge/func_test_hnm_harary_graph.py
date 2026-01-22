import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import is_isomorphic
from networkx.generators.harary_graph import hkn_harary_graph, hnm_harary_graph
def test_hnm_harary_graph(self):
    for n, m in [(5, 5), (6, 12), (7, 14)]:
        G1 = hnm_harary_graph(n, m)
        d = 2 * m // n
        G2 = nx.circulant_graph(n, list(range(1, d // 2 + 1)))
        assert is_isomorphic(G1, G2)
    for n, m in [(5, 7), (6, 13), (7, 16)]:
        G1 = hnm_harary_graph(n, m)
        d = 2 * m // n
        G2 = nx.circulant_graph(n, list(range(1, d // 2 + 1)))
        assert set(G2.edges) < set(G1.edges)
        assert G1.number_of_edges() == m
    for n, m in [(6, 9), (8, 12), (10, 15)]:
        G1 = hnm_harary_graph(n, m)
        d = 2 * m // n
        L = list(range(1, (d + 1) // 2))
        L.append(n // 2)
        G2 = nx.circulant_graph(n, L)
        assert is_isomorphic(G1, G2)
    for n, m in [(6, 10), (8, 13), (10, 17)]:
        G1 = hnm_harary_graph(n, m)
        d = 2 * m // n
        L = list(range(1, (d + 1) // 2))
        L.append(n // 2)
        G2 = nx.circulant_graph(n, L)
        assert set(G2.edges) < set(G1.edges)
        assert G1.number_of_edges() == m
    for n, m in [(5, 4), (7, 12), (9, 14)]:
        G1 = hnm_harary_graph(n, m)
        d = 2 * m // n
        L = list(range(1, (d + 1) // 2))
        G2 = nx.circulant_graph(n, L)
        assert set(G2.edges) < set(G1.edges)
        assert G1.number_of_edges() == m
    n = 0
    m = 0
    pytest.raises(nx.NetworkXError, hnm_harary_graph, n, m)
    n = 6
    m = 4
    pytest.raises(nx.NetworkXError, hnm_harary_graph, n, m)
    n = 6
    m = 16
    pytest.raises(nx.NetworkXError, hnm_harary_graph, n, m)
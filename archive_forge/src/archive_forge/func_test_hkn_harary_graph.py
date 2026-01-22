import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import is_isomorphic
from networkx.generators.harary_graph import hkn_harary_graph, hnm_harary_graph
def test_hkn_harary_graph(self):
    for k, n in [(1, 6), (1, 7)]:
        G1 = hkn_harary_graph(k, n)
        G2 = nx.path_graph(n)
        assert is_isomorphic(G1, G2)
    for k, n in [(2, 6), (2, 7), (4, 6), (4, 7)]:
        G1 = hkn_harary_graph(k, n)
        G2 = nx.circulant_graph(n, list(range(1, k // 2 + 1)))
        assert is_isomorphic(G1, G2)
    for k, n in [(3, 6), (5, 8), (7, 10)]:
        G1 = hkn_harary_graph(k, n)
        L = list(range(1, (k + 1) // 2))
        L.append(n // 2)
        G2 = nx.circulant_graph(n, L)
        assert is_isomorphic(G1, G2)
    for k, n in [(3, 5), (5, 9), (7, 11)]:
        G1 = hkn_harary_graph(k, n)
        G2 = nx.circulant_graph(n, list(range(1, (k + 1) // 2)))
        eSet1 = set(G1.edges)
        eSet2 = set(G2.edges)
        eSet3 = set()
        half = n // 2
        for i in range(half + 1):
            eSet3.add((i, (i + half) % n))
        assert eSet1 == eSet2 | eSet3
    k = 0
    n = 0
    pytest.raises(nx.NetworkXError, hkn_harary_graph, k, n)
    k = 6
    n = 6
    pytest.raises(nx.NetworkXError, hkn_harary_graph, k, n)
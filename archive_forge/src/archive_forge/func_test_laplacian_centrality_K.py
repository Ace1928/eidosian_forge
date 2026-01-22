import pytest
import networkx as nx
def test_laplacian_centrality_K():
    K = nx.krackhardt_kite_graph()
    d = nx.laplacian_centrality(K)
    exact = {0: 0.3010753, 1: 0.3010753, 2: 0.2258065, 3: 0.483871, 4: 0.2258065, 5: 0.3870968, 6: 0.3870968, 7: 0.1935484, 8: 0.0752688, 9: 0.0322581}
    for n, dc in d.items():
        assert exact[n] == pytest.approx(dc, abs=1e-07)
    full_energy = 186
    dnn = nx.laplacian_centrality(K, normalized=False)
    for n, dc in dnn.items():
        assert exact[n] * full_energy == pytest.approx(dc, abs=0.001)
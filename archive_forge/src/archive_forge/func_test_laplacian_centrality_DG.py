import pytest
import networkx as nx
def test_laplacian_centrality_DG():
    DG = nx.DiGraph([(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 6), (5, 7), (5, 8)])
    d = nx.laplacian_centrality(DG)
    exact = {0: 0.2123352, 5: 0.515391, 1: 0.2123352, 2: 0.2123352, 3: 0.2123352, 4: 0.2123352, 6: 0.2952031, 7: 0.2952031, 8: 0.2952031}
    for n, dc in d.items():
        assert exact[n] == pytest.approx(dc, abs=1e-07)
    full_energy = 9.50704
    dnn = nx.laplacian_centrality(DG, normalized=False)
    for n, dc in dnn.items():
        assert exact[n] * full_energy == pytest.approx(dc, abs=0.0001)
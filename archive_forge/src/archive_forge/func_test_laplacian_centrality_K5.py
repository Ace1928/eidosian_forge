import pytest
import networkx as nx
def test_laplacian_centrality_K5():
    K5 = nx.complete_graph(5)
    d = nx.laplacian_centrality(K5)
    exact = {0: 0.52, 1: 0.52, 2: 0.52, 3: 0.52, 4: 0.52}
    for n, dc in d.items():
        assert exact[n] == pytest.approx(dc, abs=1e-07)
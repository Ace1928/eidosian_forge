import pytest
import networkx as nx
def test_laplacian_centrality_KC():
    KC = nx.karate_club_graph()
    d = nx.laplacian_centrality(KC)
    exact = {0: 0.2543593, 1: 0.1724524, 2: 0.2166053, 3: 0.0964646, 4: 0.0350344, 5: 0.0571109, 6: 0.0540713, 7: 0.0788674, 8: 0.1222204, 9: 0.0217565, 10: 0.0308751, 11: 0.0215965, 12: 0.0174372, 13: 0.118861, 14: 0.0366341, 15: 0.0548712, 16: 0.0172772, 17: 0.0191969, 18: 0.0225564, 19: 0.0331147, 20: 0.0279955, 21: 0.0246361, 22: 0.0382339, 23: 0.1294193, 24: 0.0227164, 25: 0.0644697, 26: 0.0281555, 27: 0.075188, 28: 0.0364742, 29: 0.0707087, 30: 0.0708687, 31: 0.131019, 32: 0.2370821, 33: 0.3066709}
    for n, dc in d.items():
        assert exact[n] == pytest.approx(dc, abs=1e-07)
    full_energy = 12502
    dnn = nx.laplacian_centrality(KC, normalized=False)
    for n, dc in dnn.items():
        assert exact[n] * full_energy == pytest.approx(dc, abs=0.001)
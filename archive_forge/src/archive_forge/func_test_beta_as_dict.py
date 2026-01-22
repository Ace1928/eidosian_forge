import math
import pytest
import networkx as nx
def test_beta_as_dict(self):
    alpha = 0.1
    beta = {0: 1.0, 1: 1.0, 2: 1.0}
    b_answer = {0: 0.5598852584152165, 1: 0.6107839182711449, 2: 0.5598852584152162}
    G = nx.path_graph(3)
    b = nx.katz_centrality_numpy(G, alpha, beta)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=0.0001)
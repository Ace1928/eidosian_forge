import pytest
import networkx as nx
def test_max_deg_out_of_range():
    with pytest.raises(nx.NetworkXError, match='max_degree must be in the interval \\(0, n\\]'):
        n = 10
        tau1 = 2
        tau2 = 2
        mu = 0.1
        nx.LFR_benchmark_graph(n, tau1, tau2, mu, max_degree=n + 1, max_iters=10, seed=1)
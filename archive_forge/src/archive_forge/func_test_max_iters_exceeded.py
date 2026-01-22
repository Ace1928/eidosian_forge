import pytest
import networkx as nx
def test_max_iters_exceeded():
    with pytest.raises(nx.ExceededMaxIterations, match='Could not assign communities; try increasing min_community'):
        n = 10
        tau1 = 2
        tau2 = 2
        mu = 0.1
        nx.LFR_benchmark_graph(n, tau1, tau2, mu, min_degree=2, max_iters=10, seed=1)
import pytest
import networkx as nx
def test_max_community():
    n = 250
    tau1 = 3
    tau2 = 1.5
    mu = 0.1
    G = nx.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, max_degree=100, min_community=50, max_community=200, seed=10)
    assert len(G) == 250
    C = {frozenset(G.nodes[v]['community']) for v in G}
    assert nx.community.is_partition(G.nodes(), C)
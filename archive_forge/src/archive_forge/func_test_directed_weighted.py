import pytest
import networkx as nx
def test_directed_weighted(self):
    G = nx.DiGraph()
    G.add_edge('A', 'B', weight=5)
    G.add_edge('B', 'C', weight=1)
    G.add_edge('B', 'D', weight=0.25)
    G.add_edge('D', 'E', weight=1)
    denom = len(G) - 1
    A_local = sum([5, 3, 2.625, 2.0833333333333]) / denom
    B_local = sum([1, 0.25, 0.625]) / denom
    C_local = 0
    D_local = sum([1]) / denom
    E_local = 0
    local_reach_ctrs = [A_local, C_local, B_local, D_local, E_local]
    max_local = max(local_reach_ctrs)
    expected = sum((max_local - lrc for lrc in local_reach_ctrs)) / denom
    grc = nx.global_reaching_centrality
    actual = grc(G, normalized=False, weight='weight')
    assert expected == pytest.approx(actual, abs=1e-07)
import pytest
import networkx as nx
def test_ring_graph():
    """Second order centrality: ring graph, as defined in paper"""
    G = nx.cycle_graph(5)
    b_answer = {0: 4.472, 1: 4.472, 2: 4.472, 3: 4.472, 4: 4.472}
    b = nx.second_order_centrality(G)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=0.01)
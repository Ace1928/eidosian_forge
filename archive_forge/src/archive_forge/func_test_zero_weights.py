import pytest
import networkx as nx
def test_zero_weights(self):
    """Smoke test: ensure ZeroDivisionError is not raised."""
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=0)
    G.add_edge(0, 2, weight=0)
    S = nx.stochastic_graph(G)
    assert sorted(S.edges(data=True)) == [(0, 1, {'weight': 0}), (0, 2, {'weight': 0})]
import pytest
import networkx as nx
def test_dominating_set():
    G = nx.gnp_random_graph(100, 0.1)
    D = nx.dominating_set(G)
    assert nx.is_dominating_set(G, D)
    D = nx.dominating_set(G, start_with=0)
    assert nx.is_dominating_set(G, D)
import pytest
import networkx as nx
def test_wikipedia_is_dominating_set():
    """Example from https://en.wikipedia.org/wiki/Dominating_set"""
    G = nx.cycle_graph(4)
    G.add_edges_from([(0, 4), (1, 4), (2, 5)])
    assert nx.is_dominating_set(G, {4, 3, 5})
    assert nx.is_dominating_set(G, {0, 2})
    assert nx.is_dominating_set(G, {1, 2})
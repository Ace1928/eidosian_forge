from itertools import combinations
import pytest
import networkx as nx
def test_cyclic_graphs_raise_error():
    """
    Test that cycle graphs should cause erroring.

    This is because PGMs assume a directed acyclic graph.
    """
    g = nx.cycle_graph(3, nx.DiGraph)
    with pytest.raises(nx.NetworkXError):
        nx.d_separated(g, {0}, {1}, {2})
    with pytest.raises(nx.NetworkXError):
        nx.minimal_d_separator(g, 0, 1)
    with pytest.raises(nx.NetworkXError):
        nx.is_minimal_d_separator(g, 0, 1, {2})
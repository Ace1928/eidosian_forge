import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_asadpour_incomplete_graph():
    """
    Test that the proper exception is raised when asadpour_atsp is given an
    incomplete graph
    """
    G = nx.complete_graph(4, create_using=nx.DiGraph)
    nx.set_edge_attributes(G, 1, 'weight')
    G.remove_edge(0, 1)
    pytest.raises(nx.NetworkXError, nx_app.asadpour_atsp, G)
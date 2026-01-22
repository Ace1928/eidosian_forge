import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def test_graphs_type_exceptions():
    G = nx.Graph()
    pytest.raises(nx.NetworkXNotImplemented, nx.network_simplex, G)
    G = nx.MultiGraph()
    pytest.raises(nx.NetworkXNotImplemented, nx.network_simplex, G)
    G = nx.DiGraph()
    pytest.raises(nx.NetworkXError, nx.network_simplex, G)
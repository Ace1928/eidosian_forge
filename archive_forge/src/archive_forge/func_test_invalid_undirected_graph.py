from itertools import permutations
import pytest
import networkx as nx
def test_invalid_undirected_graph(self):
    G = nx.Graph()
    with pytest.raises(nx.NetworkXError):
        nx.average_degree_connectivity(G, target='bogus')
    with pytest.raises(nx.NetworkXError):
        nx.average_degree_connectivity(G, source='bogus')
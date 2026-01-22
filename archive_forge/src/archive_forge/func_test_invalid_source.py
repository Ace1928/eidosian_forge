from itertools import permutations
import pytest
import networkx as nx
def test_invalid_source(self):
    with pytest.raises(nx.NetworkXError):
        G = nx.DiGraph()
        nx.average_degree_connectivity(G, source='bogus')
import collections
import pytest
import networkx as nx
def test_not_eulerian(self):
    with pytest.raises(nx.NetworkXError):
        f = list(nx.eulerian_circuit(nx.complete_graph(4)))
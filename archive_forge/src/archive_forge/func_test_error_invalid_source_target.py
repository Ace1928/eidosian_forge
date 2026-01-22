import pytest
import networkx as nx
def test_error_invalid_source_target(self):
    G = nx.path_graph(4)
    with pytest.raises(nx.NetworkXError):
        nx.average_neighbor_degree(G, 'error')
    with pytest.raises(nx.NetworkXError):
        nx.average_neighbor_degree(G, 'in', 'error')
    G = G.to_directed()
    with pytest.raises(nx.NetworkXError):
        nx.average_neighbor_degree(G, 'error')
    with pytest.raises(nx.NetworkXError):
        nx.average_neighbor_degree(G, 'in', 'error')
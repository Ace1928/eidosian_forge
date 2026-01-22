import pytest
import networkx as nx
def test_weighted(self):
    G = nx.Graph()
    nx.add_cycle(G, range(7), weight=2)
    ans = nx.average_shortest_path_length(G, weight='weight')
    assert ans == pytest.approx(4, abs=1e-07)
    G = nx.Graph()
    nx.add_path(G, range(5), weight=2)
    ans = nx.average_shortest_path_length(G, weight='weight')
    assert ans == pytest.approx(4, abs=1e-07)
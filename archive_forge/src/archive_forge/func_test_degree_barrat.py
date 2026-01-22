from itertools import permutations
import pytest
import networkx as nx
def test_degree_barrat(self):
    G = nx.star_graph(5)
    G.add_edges_from([(5, 6), (5, 7), (5, 8), (5, 9)])
    G[0][5]['weight'] = 5
    nd = nx.average_degree_connectivity(G)[5]
    assert nd == 1.8
    nd = nx.average_degree_connectivity(G, weight='weight')[5]
    assert nd == pytest.approx(3.222222, abs=1e-05)
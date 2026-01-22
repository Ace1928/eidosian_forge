import pytest
import networkx as nx
def test_average_clustering_signed(self):
    G = nx.cycle_graph(3)
    G.add_edge(2, 3)
    G.add_edge(0, 1, weight=-1)
    assert nx.average_clustering(G, weight='weight') == (-1 - 1 - 1 / 3) / 4
    assert nx.average_clustering(G, weight='weight', count_zeros=True) == (-1 - 1 - 1 / 3) / 4
    assert nx.average_clustering(G, weight='weight', count_zeros=False) == (-1 - 1 - 1 / 3) / 3
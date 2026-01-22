import pytest
import networkx as nx
def test_cycle_undirected_unweighted(self):
    G = nx.Graph()
    G.add_edge(1, 2)
    assert nx.global_reaching_centrality(G, weight=None) == 0
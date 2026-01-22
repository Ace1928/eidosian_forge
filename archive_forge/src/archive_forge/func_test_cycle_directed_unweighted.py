import pytest
import networkx as nx
def test_cycle_directed_unweighted(self):
    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 1)
    assert nx.global_reaching_centrality(G, weight=None) == 0
import pytest
import networkx as nx
def test_cycle_directed_weighted(self):
    G = nx.DiGraph()
    G.add_weighted_edges_from([(1, 2, 1), (2, 1, 1)])
    assert nx.global_reaching_centrality(G) == 0
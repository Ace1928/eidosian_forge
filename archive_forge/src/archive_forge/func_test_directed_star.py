import pytest
import networkx as nx
def test_directed_star(self):
    G = nx.DiGraph()
    G.add_weighted_edges_from([(1, 2, 0.5), (1, 3, 0.5)])
    grc = nx.global_reaching_centrality
    assert grc(G, normalized=False, weight='weight') == 0.5
    assert grc(G) == 1
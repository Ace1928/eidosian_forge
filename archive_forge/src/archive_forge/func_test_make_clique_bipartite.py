import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
def test_make_clique_bipartite(self):
    G = self.G
    B = nx.make_clique_bipartite(G)
    assert sorted(B) == [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    H = nx.projected_graph(B, range(1, 12))
    assert H.adj == G.adj
    H1 = nx.projected_graph(B, range(-5, 0))
    H1 = nx.relabel_nodes(H1, {-v: v for v in range(1, 6)})
    assert sorted(H1) == [1, 2, 3, 4, 5]
import pytest
import networkx as nx
def test_p4_edge_load(self):
    G = self.P4
    c = nx.edge_load_centrality(G)
    d = {(0, 1): 6.0, (1, 2): 8.0, (2, 3): 6.0}
    for n in G.edges():
        assert c[n] == pytest.approx(d[n], abs=0.001)
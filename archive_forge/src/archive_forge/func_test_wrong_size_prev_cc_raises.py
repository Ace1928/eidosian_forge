import pytest
import networkx as nx
def test_wrong_size_prev_cc_raises(self):
    with pytest.raises(nx.NetworkXError):
        G = self.undirected_G.copy()
        edge = self.pick_add_edge(G)
        insert = True
        prev_cc = self.undirected_G_cc.copy()
        prev_cc.pop(0)
        nx.incremental_closeness_centrality(G, edge, prev_cc, insert)
import pytest
import networkx as nx
def test_weighted_load(self):
    b = nx.load_centrality(self.G, weight='weight', normalized=False)
    for n in sorted(self.G):
        assert b[n] == self.exact_weighted[n]
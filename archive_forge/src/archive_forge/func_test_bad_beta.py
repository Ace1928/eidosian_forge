import math
import pytest
import networkx as nx
def test_bad_beta(self):
    with pytest.raises(nx.NetworkXException):
        G = nx.Graph([(0, 1)])
        beta = {0: 77}
        nx.katz_centrality_numpy(G, 0.1, beta=beta)
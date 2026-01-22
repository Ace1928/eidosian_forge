import pytest
import networkx as nx
def test_normalized_p2(self):
    """
        Betweenness Centrality Subset: Normalized P2
        if n <= 2:  no normalization, betweenness centrality should be 0 for all nodes.
        """
    G = nx.Graph()
    nx.add_path(G, range(2))
    b_answer = {0: 0, 1: 0.0}
    b = nx.betweenness_centrality_subset(G, sources=[0], targets=[1], normalized=True, weight=None)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)
import pytest
import networkx as nx
from networkx import approximate_current_flow_betweenness_centrality as approximate_cfbc
from networkx import edge_current_flow_betweenness_centrality as edge_current_flow
def test_P4_normalized(self):
    """Betweenness centrality: P4 normalized"""
    G = nx.path_graph(4)
    b = nx.current_flow_betweenness_centrality(G, normalized=True)
    b_answer = {0: 0, 1: 2.0 / 3, 2: 2.0 / 3, 3: 0}
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)
import pytest
import networkx as nx
from networkx import approximate_current_flow_betweenness_centrality as approximate_cfbc
from networkx import edge_current_flow_betweenness_centrality as edge_current_flow
def test_K4_normalized(self):
    """Edge flow betweenness centrality: K4"""
    G = nx.complete_graph(4)
    b = edge_current_flow(G, normalized=False)
    b_answer = dict.fromkeys(G.edges(), 0.75)
    for (s, t), v1 in b_answer.items():
        v2 = b.get((s, t), b.get((t, s)))
        assert v1 == pytest.approx(v2, abs=1e-07)
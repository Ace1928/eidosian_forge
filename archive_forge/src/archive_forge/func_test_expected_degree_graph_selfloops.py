import pytest
import networkx as nx
def test_expected_degree_graph_selfloops():
    deg_seq = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    G1 = nx.expected_degree_graph(deg_seq, seed=1000, selfloops=False)
    G2 = nx.expected_degree_graph(deg_seq, seed=1000, selfloops=False)
    assert nx.is_isomorphic(G1, G2)
    assert len(G1) == 12
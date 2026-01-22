import pytest
import networkx as nx
def test_percolation_example1a():
    """percolation centrality: example 1a"""
    G = example1a_G()
    p = nx.percolation_centrality(G)
    p_answer = {4: 0.625, 6: 0.667}
    for n, k in p_answer.items():
        assert p[n] == pytest.approx(k, abs=0.001)
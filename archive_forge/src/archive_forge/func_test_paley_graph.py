import pytest
import networkx as nx
@pytest.mark.parametrize('p', (3, 5, 7, 11, 13))
def test_paley_graph(p):
    """Test for the :func:`networkx.paley_graph` function."""
    G = nx.paley_graph(p)
    assert len(G) == p
    in_degrees = {G.in_degree(node) for node in G.nodes}
    out_degrees = {G.out_degree(node) for node in G.nodes}
    assert len(in_degrees) == 1 and in_degrees.pop() == (p - 1) // 2
    assert len(out_degrees) == 1 and out_degrees.pop() == (p - 1) // 2
    if p % 4 == 1:
        for u, v in G.edges:
            assert (v, u) in G.edges
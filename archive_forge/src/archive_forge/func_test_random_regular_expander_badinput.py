import pytest
import networkx as nx
def test_random_regular_expander_badinput():
    pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    with pytest.raises(nx.NetworkXError, match='n must be a positive integer'):
        nx.random_regular_expander_graph(n=-1, d=2)
    with pytest.raises(nx.NetworkXError, match='d must be greater than or equal to 2'):
        nx.random_regular_expander_graph(n=10, d=0)
    with pytest.raises(nx.NetworkXError, match='Need n-1>= d to have room'):
        nx.random_regular_expander_graph(n=5, d=6)
    with pytest.raises(nx.NetworkXError, match='epsilon must be non negative'):
        nx.random_regular_expander_graph(n=4, d=2, epsilon=-1)
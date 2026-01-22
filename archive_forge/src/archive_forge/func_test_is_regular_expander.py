import pytest
import networkx as nx
@pytest.mark.parametrize('n', (3, 5, 6, 10))
def test_is_regular_expander(n):
    pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    G = nx.complete_graph(n)
    assert nx.is_regular_expander(G) == True, 'Should be a regular expander'
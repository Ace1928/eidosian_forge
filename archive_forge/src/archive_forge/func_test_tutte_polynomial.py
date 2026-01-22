import pytest
import networkx as nx
@pytest.mark.parametrize(('G', 'expected'), _test_tutte_graphs.items())
def test_tutte_polynomial(G, expected):
    assert nx.tutte_polynomial(G).equals(expected)
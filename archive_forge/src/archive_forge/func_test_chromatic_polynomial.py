import pytest
import networkx as nx
@pytest.mark.parametrize(('G', 'expected'), _test_chromatic_graphs.items())
def test_chromatic_polynomial(G, expected):
    assert nx.chromatic_polynomial(G).equals(expected)
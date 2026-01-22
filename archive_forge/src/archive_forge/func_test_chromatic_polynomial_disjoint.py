import pytest
import networkx as nx
@pytest.mark.parametrize('G', _test_chromatic_graphs.keys())
def test_chromatic_polynomial_disjoint(G):
    """Chromatic polynomial factors into the Chromatic polynomials of its
    components. Verify this property with the disjoint union of two copies of
    the input graph.
    """
    x_g = nx.chromatic_polynomial(G)
    H = nx.disjoint_union(G, G)
    x_h = nx.chromatic_polynomial(H)
    assert sympy.simplify(x_g * x_g).equals(x_h)
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import factor
from sympy.core import symbols
from sympy.tensor.indexed import IndexedBase
from sympy.polys.multivariate_resultants import (DixonResultant,
def test_get_dixon_polynomial_numerical():
    """Test Dixon's polynomial for a numerical example."""
    a = IndexedBase('alpha')
    p = x + y
    q = x ** 2 + y ** 3
    h = x ** 2 + y
    dixon = DixonResultant([p, q, h], [x, y])
    polynomial = -x * y ** 2 * a[0] - x * y ** 2 * a[1] - x * y * a[0] * a[1] - x * y * a[1] ** 2 - x * a[0] * a[1] ** 2 + x * a[0] - y ** 2 * a[0] * a[1] + y ** 2 * a[1] - y * a[0] * a[1] ** 2 + y * a[1] ** 2
    assert dixon.get_dixon_polynomial().as_expr().expand() == polynomial
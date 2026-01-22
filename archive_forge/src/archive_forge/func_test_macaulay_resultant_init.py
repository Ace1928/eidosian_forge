from sympy.matrices.dense import Matrix
from sympy.polys.polytools import factor
from sympy.core import symbols
from sympy.tensor.indexed import IndexedBase
from sympy.polys.multivariate_resultants import (DixonResultant,
def test_macaulay_resultant_init():
    """Test init method of MacaulayResultant."""
    assert macaulay.polynomials == [p, q]
    assert macaulay.variables == [x, y]
    assert macaulay.n == 2
    assert macaulay.degrees == [1, 1]
    assert macaulay.degree_m == 1
    assert macaulay.monomials_size == 2
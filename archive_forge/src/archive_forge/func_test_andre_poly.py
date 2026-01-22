from sympy.core.numbers import Rational as Q
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
from sympy.polys.appellseqs import (bernoulli_poly, bernoulli_c_poly,
from sympy.abc import x
def test_andre_poly():
    raises(ValueError, lambda: andre_poly(-1, x))
    assert andre_poly(1, x, polys=True) == Poly(x)
    assert andre_poly(0, x) == 1
    assert andre_poly(1, x) == x
    assert andre_poly(2, x) == x ** 2 - 1
    assert andre_poly(3, x) == x ** 3 - 3 * x
    assert andre_poly(4, x) == x ** 4 - 6 * x ** 2 + 5
    assert andre_poly(5, x) == x ** 5 - 10 * x ** 3 + 25 * x
    assert andre_poly(6, x) == x ** 6 - 15 * x ** 4 + 75 * x ** 2 - 61
    assert andre_poly(1).dummy_eq(x)
    assert andre_poly(1, polys=True) == Poly(x)
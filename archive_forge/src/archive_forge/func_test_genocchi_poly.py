from sympy.core.numbers import Rational as Q
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
from sympy.polys.appellseqs import (bernoulli_poly, bernoulli_c_poly,
from sympy.abc import x
def test_genocchi_poly():
    raises(ValueError, lambda: genocchi_poly(-1, x))
    assert genocchi_poly(2, x, polys=True) == Poly(-2 * x + 1)
    assert genocchi_poly(0, x) == 0
    assert genocchi_poly(1, x) == -1
    assert genocchi_poly(2, x) == 1 - 2 * x
    assert genocchi_poly(3, x) == 3 * x - 3 * x ** 2
    assert genocchi_poly(4, x) == -1 + 6 * x ** 2 - 4 * x ** 3
    assert genocchi_poly(5, x) == -5 * x + 10 * x ** 3 - 5 * x ** 4
    assert genocchi_poly(6, x) == 3 - 15 * x ** 2 + 15 * x ** 4 - 6 * x ** 5
    assert genocchi_poly(2).dummy_eq(-2 * x + 1)
    assert genocchi_poly(2, polys=True) == Poly(-2 * x + 1)
    assert 2 * (bernoulli_poly(8, x) - bernoulli_c_poly(8, x)) == genocchi_poly(8, x)
    assert 2 * (bernoulli_poly(9, x) - bernoulli_c_poly(9, x)) == genocchi_poly(9, x)
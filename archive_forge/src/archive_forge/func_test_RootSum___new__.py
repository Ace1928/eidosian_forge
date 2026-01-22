from sympy.polys.polytools import Poly
import sympy.polys.rootoftools as rootoftools
from sympy.polys.rootoftools import (rootof, RootOf, CRootOf, RootSum,
from sympy.polys.polyerrors import (
from sympy.core.function import (Function, Lambda)
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import tan
from sympy.integrals.integrals import Integral
from sympy.polys.orthopolys import legendre_poly
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises, slow
from sympy.core.expr import unchanged
from sympy.abc import a, b, x, y, z, r
def test_RootSum___new__():
    f = x ** 3 + x + 3
    g = Lambda(r, log(r * x))
    s = RootSum(f, g)
    assert isinstance(s, RootSum) is True
    assert RootSum(f ** 2, g) == 2 * RootSum(f, g)
    assert RootSum((x - 7) * f ** 3, g) == log(7 * x) + 3 * RootSum(f, g)
    assert hash(RootSum((x - 7) * f ** 3, g)) == hash(log(7 * x) + 3 * RootSum(f, g))
    raises(MultivariatePolynomialError, lambda: RootSum(x ** 3 + x + y))
    raises(ValueError, lambda: RootSum(x ** 2 + 3, lambda x: x))
    assert RootSum(f, exp) == RootSum(f, Lambda(x, exp(x)))
    assert RootSum(f, log) == RootSum(f, Lambda(x, log(x)))
    assert isinstance(RootSum(f, auto=False), RootSum) is True
    assert RootSum(f) == 0
    assert RootSum(f, Lambda(x, x)) == 0
    assert RootSum(f, Lambda(x, x ** 2)) == -2
    assert RootSum(f, Lambda(x, 1)) == 3
    assert RootSum(f, Lambda(x, 2)) == 6
    assert RootSum(f, auto=False).is_commutative is True
    assert RootSum(f, Lambda(x, 1 / (x + x ** 2))) == Rational(11, 3)
    assert RootSum(f, Lambda(x, y / (x + x ** 2))) == Rational(11, 3) * y
    assert RootSum(x ** 2 - 1, Lambda(x, 3 * x ** 2), x) == 6
    assert RootSum(x ** 2 - y, Lambda(x, 3 * x ** 2), x) == 6 * y
    assert RootSum(x ** 2 - 1, Lambda(x, z * x ** 2), x) == 2 * z
    assert RootSum(x ** 2 - y, Lambda(x, z * x ** 2), x) == 2 * z * y
    assert RootSum(x ** 2 - 1, Lambda(x, exp(x)), quadratic=True) == exp(-1) + exp(1)
    assert RootSum(x ** 3 + a * x + a ** 3, tan, x) == RootSum(x ** 3 + x + 1, Lambda(x, tan(a * x)))
    assert RootSum(a ** 3 * x ** 3 + a * x + 1, tan, x) == RootSum(x ** 3 + x + 1, Lambda(x, tan(x / a)))
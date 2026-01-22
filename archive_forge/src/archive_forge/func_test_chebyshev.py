from sympy.concrete.summations import Sum
from sympy.core.function import (Derivative, diff)
from sympy.core.numbers import (Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import (RisingFactorial, binomial, factorial)
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.functions.special.polynomials import (assoc_laguerre, assoc_legendre, chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root, gegenbauer, hermite, hermite_prob, jacobi, jacobi_normalized, laguerre, legendre)
from sympy.polys.orthopolys import laguerre_poly
from sympy.polys.polyroots import roots
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises
def test_chebyshev():
    assert chebyshevt(0, x) == 1
    assert chebyshevt(1, x) == x
    assert chebyshevt(2, x) == 2 * x ** 2 - 1
    assert chebyshevt(3, x) == 4 * x ** 3 - 3 * x
    for n in range(1, 4):
        for k in range(n):
            z = chebyshevt_root(n, k)
            assert chebyshevt(n, z) == 0
        raises(ValueError, lambda: chebyshevt_root(n, n))
    for n in range(1, 4):
        for k in range(n):
            z = chebyshevu_root(n, k)
            assert chebyshevu(n, z) == 0
        raises(ValueError, lambda: chebyshevu_root(n, n))
    n = Symbol('n')
    X = chebyshevt(n, x)
    assert isinstance(X, chebyshevt)
    assert unchanged(chebyshevt, n, x)
    assert chebyshevt(n, -x) == (-1) ** n * chebyshevt(n, x)
    assert chebyshevt(-n, x) == chebyshevt(n, x)
    assert chebyshevt(n, 0) == cos(pi * n / 2)
    assert chebyshevt(n, 1) == 1
    assert chebyshevt(n, oo) is oo
    assert conjugate(chebyshevt(n, x)) == chebyshevt(n, conjugate(x))
    assert diff(chebyshevt(n, x), x) == n * chebyshevu(n - 1, x)
    X = chebyshevu(n, x)
    assert isinstance(X, chebyshevu)
    y = Symbol('y')
    assert chebyshevu(n, -x) == (-1) ** n * chebyshevu(n, x)
    assert chebyshevu(-n, x) == -chebyshevu(n - 2, x)
    assert unchanged(chebyshevu, -n + y, x)
    assert chebyshevu(n, 0) == cos(pi * n / 2)
    assert chebyshevu(n, 1) == n + 1
    assert chebyshevu(n, oo) is oo
    assert conjugate(chebyshevu(n, x)) == chebyshevu(n, conjugate(x))
    assert diff(chebyshevu(n, x), x) == (-x * chebyshevu(n, x) + (n + 1) * chebyshevt(n + 1, x)) / (x ** 2 - 1)
    _k = Dummy('k')
    assert chebyshevt(n, x).rewrite(Sum).dummy_eq(Sum(x ** (-2 * _k + n) * (x ** 2 - 1) ** _k * binomial(n, 2 * _k), (_k, 0, floor(n / 2))))
    assert chebyshevt(n, x).rewrite('polynomial').dummy_eq(Sum(x ** (-2 * _k + n) * (x ** 2 - 1) ** _k * binomial(n, 2 * _k), (_k, 0, floor(n / 2))))
    assert chebyshevu(n, x).rewrite(Sum).dummy_eq(Sum((-1) ** _k * (2 * x) ** (-2 * _k + n) * factorial(-_k + n) / (factorial(_k) * factorial(-2 * _k + n)), (_k, 0, floor(n / 2))))
    assert chebyshevu(n, x).rewrite('polynomial').dummy_eq(Sum((-1) ** _k * (2 * x) ** (-2 * _k + n) * factorial(-_k + n) / (factorial(_k) * factorial(-2 * _k + n)), (_k, 0, floor(n / 2))))
    raises(ArgumentIndexError, lambda: chebyshevt(n, x).fdiff(1))
    raises(ArgumentIndexError, lambda: chebyshevt(n, x).fdiff(3))
    raises(ArgumentIndexError, lambda: chebyshevu(n, x).fdiff(1))
    raises(ArgumentIndexError, lambda: chebyshevu(n, x).fdiff(3))
from sympy.core.function import (Function, Lambda, diff, expand_log)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, tan)
from sympy.polys.polytools import (Poly, cancel, factor)
from sympy.integrals.risch import (gcdex_diophantine, frac_in, as_poly_1t,
from sympy.testing.pytest import raises
from sympy.abc import x, t, nu, z, a, y
def test_DifferentialExtension_misc():
    assert DifferentialExtension(sin(y) * exp(x), x)._important_attrs == (Poly(sin(y) * t0, t0, domain='ZZ[sin(y)]'), Poly(1, t0, domain='ZZ'), [Poly(1, x, domain='ZZ'), Poly(t0, t0, domain='ZZ')], [x, t0], [Lambda(i, exp(i))], [], [None, 'exp'], [None, x])
    raises(NotImplementedError, lambda: DifferentialExtension(sin(x), x))
    assert DifferentialExtension(10 ** x, x)._important_attrs == (Poly(t0, t0), Poly(1, t0), [Poly(1, x), Poly(log(10) * t0, t0)], [x, t0], [Lambda(i, exp(i * log(10)))], [(exp(x * log(10)), 10 ** x)], [None, 'exp'], [None, x * log(10)])
    assert DifferentialExtension(log(x) + log(x ** 2), x)._important_attrs in [(Poly(3 * t0, t0), Poly(2, t0), [Poly(1, x), Poly(2 / x, t0)], [x, t0], [Lambda(i, log(i ** 2))], [], [None], [], [1], [x ** 2]), (Poly(3 * t0, t0), Poly(1, t0), [Poly(1, x), Poly(1 / x, t0)], [x, t0], [Lambda(i, log(i))], [], [None, 'log'], [None, x])]
    assert DifferentialExtension(S.Zero, x)._important_attrs == (Poly(0, x), Poly(1, x), [Poly(1, x)], [x], [], [], [None], [None])
    assert DifferentialExtension(tan(atan(x).rewrite(log)), x)._important_attrs == (Poly(x, x), Poly(1, x), [Poly(1, x)], [x], [], [], [None], [None])
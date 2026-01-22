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
def test_frac_in():
    assert frac_in(Poly((x + 1) / x * t, t), x) == (Poly(t * x + t, x), Poly(x, x))
    assert frac_in((x + 1) / x * t, x) == (Poly(t * x + t, x), Poly(x, x))
    assert frac_in((Poly((x + 1) / x * t, t), Poly(t + 1, t)), x) == (Poly(t * x + t, x), Poly((1 + t) * x, x))
    raises(ValueError, lambda: frac_in((x + 1) / log(x) * t, x))
    assert frac_in(Poly((2 + 2 * x + x * (1 + x)) / (1 + x) ** 2, t), x, cancel=True) == (Poly(x + 2, x), Poly(x + 1, x))
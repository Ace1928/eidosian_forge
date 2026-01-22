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
def test_gcdex_diophantine():
    assert gcdex_diophantine(Poly(x ** 4 - 2 * x ** 3 - 6 * x ** 2 + 12 * x + 15), Poly(x ** 3 + x ** 2 - 4 * x - 4), Poly(x ** 2 - 1)) == (Poly((-x ** 2 + 4 * x - 3) / 5), Poly((x ** 3 - 7 * x ** 2 + 16 * x - 10) / 5))
    assert gcdex_diophantine(Poly(x ** 3 + 6 * x + 7), Poly(x ** 2 + 3 * x + 2), Poly(x + 1)) == (Poly(1 / 13, x, domain='QQ'), Poly(-1 / 13 * x + 3 / 13, x, domain='QQ'))
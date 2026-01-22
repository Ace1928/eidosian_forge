from sympy.core.numbers import (I, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (sin, tan)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.integrals.integrals import Integral
from sympy.series.order import O
from sympy.functions.special.elliptic_integrals import (elliptic_k as K,
from sympy.core.random import (test_derivative_numerically as td,
from sympy.abc import z, m, n
def test_F():
    assert F(z, 0) == z
    assert F(0, m) == 0
    assert F(pi * i / 2, m) == i * K(m)
    assert F(z, oo) == 0
    assert F(z, -oo) == 0
    assert F(-z, m) == -F(z, m)
    assert F(z, m).diff(z) == 1 / sqrt(1 - m * sin(z) ** 2)
    assert F(z, m).diff(m) == E(z, m) / (2 * m * (1 - m)) - F(z, m) / (2 * m) - sin(2 * z) / (4 * (1 - m) * sqrt(1 - m * sin(z) ** 2))
    r = randcplx()
    assert td(F(z, r), z)
    assert td(F(r, m), m)
    mi = Symbol('m', real=False)
    assert F(z, mi).conjugate() == F(z.conjugate(), mi.conjugate())
    mr = Symbol('m', negative=True)
    assert F(z, mr).conjugate() == F(z.conjugate(), mr)
    assert F(z, m).series(z) == z + z ** 5 * (3 * m ** 2 / 40 - m / 30) + m * z ** 3 / 6 + O(z ** 6)
    assert F(z, m).rewrite(Integral).dummy_eq(Integral(1 / sqrt(1 - m * sin(t) ** 2), (t, 0, z)))
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
def test_K():
    assert K(0) == pi / 2
    assert K(S.Half) == 8 * pi ** Rational(3, 2) / gamma(Rational(-1, 4)) ** 2
    assert K(1) is zoo
    assert K(-1) == gamma(Rational(1, 4)) ** 2 / (4 * sqrt(2 * pi))
    assert K(oo) == 0
    assert K(-oo) == 0
    assert K(I * oo) == 0
    assert K(-I * oo) == 0
    assert K(zoo) == 0
    assert K(z).diff(z) == (E(z) - (1 - z) * K(z)) / (2 * z * (1 - z))
    assert td(K(z), z)
    zi = Symbol('z', real=False)
    assert K(zi).conjugate() == K(zi.conjugate())
    zr = Symbol('z', negative=True)
    assert K(zr).conjugate() == K(zr)
    assert K(z).rewrite(hyper) == pi / 2 * hyper((S.Half, S.Half), (S.One,), z)
    assert tn(K(z), pi / 2 * hyper((S.Half, S.Half), (S.One,), z))
    assert K(z).rewrite(meijerg) == meijerg(((S.Half, S.Half), []), ((S.Zero,), (S.Zero,)), -z) / 2
    assert tn(K(z), meijerg(((S.Half, S.Half), []), ((S.Zero,), (S.Zero,)), -z) / 2)
    assert K(z).series(z) == pi / 2 + pi * z / 8 + 9 * pi * z ** 2 / 128 + 25 * pi * z ** 3 / 512 + 1225 * pi * z ** 4 / 32768 + 3969 * pi * z ** 5 / 131072 + O(z ** 6)
    assert K(m).rewrite(Integral).dummy_eq(Integral(1 / sqrt(1 - m * sin(t) ** 2), (t, 0, pi / 2)))
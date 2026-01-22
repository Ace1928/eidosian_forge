from sympy.core.function import (diff, expand, expand_func)
from sympy.core import EulerGamma
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (conjugate, im, polar_lift, re)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.functions.special.error_functions import (Chi, Ci, E1, Ei, Li, Shi, Si, erf, erf2, erf2inv, erfc, erfcinv, erfi, erfinv, expint, fresnelc, fresnels, li)
from sympy.functions.special.gamma_functions import (gamma, uppergamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.series.gruntz import gruntz
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.functions.special.error_functions import _erfs, _eis
from sympy.testing.pytest import raises
def test_li():
    z = Symbol('z')
    zr = Symbol('z', real=True)
    zp = Symbol('z', positive=True)
    zn = Symbol('z', negative=True)
    assert li(0) is S.Zero
    assert li(1) is -oo
    assert li(oo) is oo
    assert isinstance(li(z), li)
    assert unchanged(li, -zp)
    assert unchanged(li, zn)
    assert diff(li(z), z) == 1 / log(z)
    assert conjugate(li(z)) == li(conjugate(z))
    assert conjugate(li(-zr)) == li(-zr)
    assert unchanged(conjugate, li(-zp))
    assert unchanged(conjugate, li(zn))
    assert li(z).rewrite(Li) == Li(z) + li(2)
    assert li(z).rewrite(Ei) == Ei(log(z))
    assert li(z).rewrite(uppergamma) == -log(1 / log(z)) / 2 - log(-log(z)) + log(log(z)) / 2 - expint(1, -log(z))
    assert li(z).rewrite(Si) == -log(I * log(z)) - log(1 / log(z)) / 2 + log(log(z)) / 2 + Ci(I * log(z)) + Shi(log(z))
    assert li(z).rewrite(Ci) == -log(I * log(z)) - log(1 / log(z)) / 2 + log(log(z)) / 2 + Ci(I * log(z)) + Shi(log(z))
    assert li(z).rewrite(Shi) == -log(1 / log(z)) / 2 + log(log(z)) / 2 + Chi(log(z)) - Shi(log(z))
    assert li(z).rewrite(Chi) == -log(1 / log(z)) / 2 + log(log(z)) / 2 + Chi(log(z)) - Shi(log(z))
    assert li(z).rewrite(hyper) == log(z) * hyper((1, 1), (2, 2), log(z)) - log(1 / log(z)) / 2 + log(log(z)) / 2 + EulerGamma
    assert li(z).rewrite(meijerg) == -log(1 / log(z)) / 2 - log(-log(z)) + log(log(z)) / 2 - meijerg(((), (1,)), ((0, 0), ()), -log(z))
    assert gruntz(1 / li(z), z, oo) is S.Zero
    assert li(z).series(z) == log(z) ** 5 / 600 + log(z) ** 4 / 96 + log(z) ** 3 / 18 + log(z) ** 2 / 4 + log(z) + log(log(z)) + EulerGamma
    raises(ArgumentIndexError, lambda: li(z).fdiff(2))
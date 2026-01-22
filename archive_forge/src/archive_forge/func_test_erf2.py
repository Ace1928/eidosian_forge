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
def test_erf2():
    assert erf2(0, 0) is S.Zero
    assert erf2(x, x) is S.Zero
    assert erf2(nan, 0) is nan
    assert erf2(-oo, y) == erf(y) + 1
    assert erf2(oo, y) == erf(y) - 1
    assert erf2(x, oo) == 1 - erf(x)
    assert erf2(x, -oo) == -1 - erf(x)
    assert erf2(x, erf2inv(x, y)) == y
    assert erf2(-x, -y) == -erf2(x, y)
    assert erf2(-x, y) == erf(y) + erf(x)
    assert erf2(x, -y) == -erf(y) - erf(x)
    assert erf2(x, y).rewrite('fresnels') == erf(y).rewrite(fresnels) - erf(x).rewrite(fresnels)
    assert erf2(x, y).rewrite('fresnelc') == erf(y).rewrite(fresnelc) - erf(x).rewrite(fresnelc)
    assert erf2(x, y).rewrite('hyper') == erf(y).rewrite(hyper) - erf(x).rewrite(hyper)
    assert erf2(x, y).rewrite('meijerg') == erf(y).rewrite(meijerg) - erf(x).rewrite(meijerg)
    assert erf2(x, y).rewrite('uppergamma') == erf(y).rewrite(uppergamma) - erf(x).rewrite(uppergamma)
    assert erf2(x, y).rewrite('expint') == erf(y).rewrite(expint) - erf(x).rewrite(expint)
    assert erf2(I, 0).is_real is False
    assert erf2(0, 0, evaluate=False).is_real
    assert erf2(0, 0, evaluate=False).is_zero
    assert erf2(x, x, evaluate=False).is_zero
    assert erf2(x, y).is_zero is None
    assert expand_func(erf(x) + erf2(x, y)) == erf(y)
    assert conjugate(erf2(x, y)) == erf2(conjugate(x), conjugate(y))
    assert erf2(x, y).rewrite('erf') == erf(y) - erf(x)
    assert erf2(x, y).rewrite('erfc') == erfc(x) - erfc(y)
    assert erf2(x, y).rewrite('erfi') == I * (erfi(I * x) - erfi(I * y))
    assert erf2(x, y).diff(x) == erf2(x, y).fdiff(1)
    assert erf2(x, y).diff(y) == erf2(x, y).fdiff(2)
    assert erf2(x, y).diff(x) == -2 * exp(-x ** 2) / sqrt(pi)
    assert erf2(x, y).diff(y) == 2 * exp(-y ** 2) / sqrt(pi)
    raises(ArgumentIndexError, lambda: erf2(x, y).fdiff(3))
    assert erf2(x, y).is_extended_real is None
    xr, yr = symbols('xr yr', extended_real=True)
    assert erf2(xr, yr).is_extended_real is True
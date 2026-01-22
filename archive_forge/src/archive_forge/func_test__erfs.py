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
def test__erfs():
    assert _erfs(z).diff(z) == -2 / sqrt(S.Pi) + 2 * z * _erfs(z)
    assert _erfs(1 / z).series(z) == z / sqrt(pi) - z ** 3 / (2 * sqrt(pi)) + 3 * z ** 5 / (4 * sqrt(pi)) + O(z ** 6)
    assert expand(erf(z).rewrite('tractable').diff(z).rewrite('intractable')) == erf(z).diff(z)
    assert _erfs(z).rewrite('intractable') == (-erf(z) + 1) * exp(z ** 2)
    raises(ArgumentIndexError, lambda: _erfs(z).fdiff(2))
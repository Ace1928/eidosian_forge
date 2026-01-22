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
def test_Li():
    assert Li(2) is S.Zero
    assert Li(oo) is oo
    assert isinstance(Li(z), Li)
    assert diff(Li(z), z) == 1 / log(z)
    assert gruntz(1 / Li(z), z, oo) is S.Zero
    assert Li(z).rewrite(li) == li(z) - li(2)
    assert Li(z).series(z) == log(z) ** 5 / 600 + log(z) ** 4 / 96 + log(z) ** 3 / 18 + log(z) ** 2 / 4 + log(z) + log(log(z)) - li(2) + EulerGamma
    raises(ArgumentIndexError, lambda: Li(z).fdiff(2))
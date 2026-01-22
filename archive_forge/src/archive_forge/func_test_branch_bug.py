from sympy.core.function import expand_func
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.complexes import Abs, arg, re, unpolarify
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import cosh, acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (cos, sin, sinc, asin)
from sympy.functions.special.error_functions import (erf, erfc)
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import simplify
from sympy.integrals.meijerint import (_rewrite_single, _rewrite1,
from sympy.testing.pytest import slow
from sympy.core.random import (verify_numerically,
from sympy.abc import x, y, a, b, c, d, s, t, z
def test_branch_bug():
    from sympy.functions.special.gamma_functions import lowergamma
    from sympy.simplify.powsimp import powdenest
    assert powdenest(integrate(erf(x ** 3), x, meijerg=True).diff(x), polar=True) == 2 * erf(x ** 3) * gamma(Rational(2, 3)) / 3 / gamma(Rational(5, 3))
    assert integrate(erf(x ** 3), x, meijerg=True) == 2 * x * erf(x ** 3) * gamma(Rational(2, 3)) / (3 * gamma(Rational(5, 3))) - 2 * gamma(Rational(2, 3)) * lowergamma(Rational(2, 3), x ** 6) / (3 * sqrt(pi) * gamma(Rational(5, 3)))
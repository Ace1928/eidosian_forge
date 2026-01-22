from sympy.holonomic import (DifferentialOperator, HolonomicFunction,
from sympy.holonomic.recurrence import RecurrenceOperators, HolonomicSequence
from sympy.core import EulerGamma
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.bessel import besselj
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.error_functions import (Ci, Si, erf, erfc)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.simplify.hyperexpand import hyperexpand
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.realfield import RR
def test_from_hyper():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = hyper([1, 1], [Rational(3, 2)], x ** 2 / 4)
    q = HolonomicFunction(4 * x + (5 * x ** 2 - 8) * Dx + (x ** 3 - 4 * x) * Dx ** 2, x, 1, [2 * sqrt(3) * pi / 9, -4 * sqrt(3) * pi / 27 + Rational(4, 3)])
    r = from_hyper(p)
    assert r == q
    p = from_hyper(hyper([1], [Rational(3, 2)], x ** 2 / 4))
    q = HolonomicFunction(-x + (-x ** 2 / 2 + 2) * Dx + x * Dx ** 2, x)
    y0 = '[sqrt(pi)*exp(1/4)*erf(1/2), -sqrt(pi)*exp(1/4)*erf(1/2)/2 + 1]'
    assert sstr(p.y0) == y0
    assert q.annihilator == p.annihilator
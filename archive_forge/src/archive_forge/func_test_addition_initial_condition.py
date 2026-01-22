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
def test_addition_initial_condition():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction(Dx - 1, x, 0, [3])
    q = HolonomicFunction(Dx ** 2 + 1, x, 0, [1, 0])
    r = HolonomicFunction(-1 + Dx - Dx ** 2 + Dx ** 3, x, 0, [4, 3, 2])
    assert p + q == r
    p = HolonomicFunction(Dx - x + Dx ** 2, x, 0, [1, 2])
    q = HolonomicFunction(Dx ** 2 + x, x, 0, [1, 0])
    r = HolonomicFunction(-x ** 4 - x ** 3 / 4 - x ** 2 + Rational(1, 4) + (x ** 3 + x ** 2 / 4 + x * Rational(3, 4) + 1) * Dx + (x * Rational(-3, 2) + Rational(7, 4)) * Dx ** 2 + (x ** 2 - x * Rational(7, 4) + Rational(1, 4)) * Dx ** 3 + (x ** 2 + x / 4 + S.Half) * Dx ** 4, x, 0, [2, 2, -2, 2])
    assert p + q == r
    p = HolonomicFunction(Dx ** 2 + 4 * x * Dx + x ** 2, x, 0, [3, 4])
    q = HolonomicFunction(Dx ** 2 + 1, x, 0, [1, 1])
    r = HolonomicFunction(x ** 6 + 2 * x ** 4 - 5 * x ** 2 - 6 + (4 * x ** 5 + 36 * x ** 3 - 32 * x) * Dx + (x ** 6 + 3 * x ** 4 + 5 * x ** 2 - 9) * Dx ** 2 + (4 * x ** 5 + 36 * x ** 3 - 32 * x) * Dx ** 3 + (x ** 4 + 10 * x ** 2 - 3) * Dx ** 4, x, 0, [4, 5, -1, -17])
    assert p + q == r
    q = HolonomicFunction(Dx ** 3 + x, x, 2, [3, 0, 1])
    p = HolonomicFunction(Dx - 1, x, 2, [1])
    r = HolonomicFunction(-x ** 2 - x + 1 + (x ** 2 + x) * Dx + (-x - 2) * Dx ** 3 + (x + 1) * Dx ** 4, x, 2, [4, 1, 2, -5])
    assert p + q == r
    p = expr_to_holonomic(sin(x))
    q = expr_to_holonomic(1 / x, x0=1)
    r = HolonomicFunction(x ** 2 + 6 + (x ** 3 + 2 * x) * Dx + (x ** 2 + 6) * Dx ** 2 + (x ** 3 + 2 * x) * Dx ** 3, x, 1, [sin(1) + 1, -1 + cos(1), -sin(1) + 2])
    assert p + q == r
    C_1 = symbols('C_1')
    p = expr_to_holonomic(sqrt(x))
    q = expr_to_holonomic(sqrt(x ** 2 - x))
    r = (p + q).to_expr().subs(C_1, -I / 2).expand()
    assert r == I * sqrt(x) * sqrt(-x + 1) + sqrt(x)
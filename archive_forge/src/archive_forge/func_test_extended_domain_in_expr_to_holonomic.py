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
def test_extended_domain_in_expr_to_holonomic():
    x = symbols('x')
    p = expr_to_holonomic(1.2 * cos(3.1 * x))
    assert p.to_expr() == 1.2 * cos(3.1 * x)
    assert sstr(p.integrate(x).to_expr()) == '0.387096774193548*sin(3.1*x)'
    _, Dx = DifferentialOperators(RR.old_poly_ring(x), 'Dx')
    p = expr_to_holonomic(1.1329138213 * x)
    q = HolonomicFunction(-1.1329138213 + 1.1329138213 * x * Dx, x, 0, {1: [1.1329138213]})
    assert p == q
    assert p.to_expr() == 1.1329138213 * x
    assert sstr(p.integrate((x, 1, 2))) == sstr((1.1329138213 * x).integrate((x, 1, 2)))
    y, z = symbols('y, z')
    p = expr_to_holonomic(sin(x * y * z), x=x)
    assert p.to_expr() == sin(x * y * z)
    assert p.integrate(x).to_expr() == (-cos(x * y * z) + 1) / (y * z)
    p = expr_to_holonomic(sin(x * y + z), x=x).integrate(x).to_expr()
    q = (cos(z) - cos(x * y + z)) / y
    assert p == q
    a = symbols('a')
    p = expr_to_holonomic(a * x, x)
    assert p.to_expr() == a * x
    assert p.integrate(x).to_expr() == a * x ** 2 / 2
    D_2, C_1 = symbols('D_2, C_1')
    p = expr_to_holonomic(x) + expr_to_holonomic(1.2 * cos(x))
    p = p.to_expr().subs(D_2, 0)
    assert p - x - 1.2 * cos(1.0 * x) == 0
    p = expr_to_holonomic(x) * expr_to_holonomic(1.2 * cos(x))
    p = p.to_expr().subs(C_1, 0)
    assert p - 1.2 * x * cos(1.0 * x) == 0
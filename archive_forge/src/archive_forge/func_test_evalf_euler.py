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
def test_evalf_euler():
    x = symbols('x')
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')
    p = HolonomicFunction((1 + x) * Dx ** 2 + Dx, x, 0, [0, 1])
    r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    s = '0.699525841805253'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s
    r = [0.1 + 0.1 * I]
    for i in range(9):
        r.append(r[-1] + 0.1 + 0.1 * I)
    for i in range(10):
        r.append(r[-1] + 0.1 - 0.1 * I)
    s = '1.07530466271334 - 0.0251200594793912*I'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s
    p = HolonomicFunction(Dx ** 2 + 1, x, 0, [0, 1])
    s = '0.905546532085401 - 6.93889390390723e-18*I'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s
    r = [0.1]
    for i in range(14):
        r.append(r[-1] + 0.1)
    r.append(pi / 2)
    s = '1.08016557252834'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s
    r = [0.1 * I]
    for i in range(9):
        r.append(r[-1] + 0.1 * I)
    for i in range(15):
        r.append(r[-1] + 0.1)
    r.append(pi / 2 + I)
    for i in range(10):
        r.append(r[-1] - 0.1 * I)
    s = '0.976882381836257 - 1.65557671738537e-16*I'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s
    p = HolonomicFunction(Dx ** 2 + 1, x, 0, [1, 0])
    r = [0.05]
    for i in range(61):
        r.append(r[-1] + 0.05)
    r.append(pi)
    s = '-1.08140824719196'
    assert sstr(p.evalf(r, method='Euler')[-1]) == s
    r = [0.1 * I]
    for i in range(9):
        r.append(r[-1] + 0.1 * I)
    for i in range(20):
        r.append(r[-1] + 0.1)
    for i in range(10):
        r.append(r[-1] - 0.1 * I)
    p = HolonomicFunction(Dx ** 2 + 1, x, 0, [1, 1]).evalf(r, method='Euler')
    s = '0.501421652861245 - 3.88578058618805e-16*I'
    assert sstr(p[-1]) == s
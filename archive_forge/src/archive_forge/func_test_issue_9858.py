from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.function import (Derivative, Function, diff, expand)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, csch, cosh, coth, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, cos, cot, csc, sec, sin, tan)
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.functions.special.elliptic_integrals import (elliptic_e, elliptic_f)
from sympy.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, erf, erfi, fresnelc, fresnels, li)
from sympy.functions.special.gamma_functions import uppergamma
from sympy.functions.special.polynomials import (assoc_laguerre, chebyshevt, chebyshevu, gegenbauer, hermite, jacobi, laguerre, legendre)
from sympy.functions.special.zeta_functions import polylog
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import And
from sympy.integrals.manualintegrate import (manualintegrate, find_substitutions,
from sympy.testing.pytest import raises, slow
@slow
def test_issue_9858():
    assert manualintegrate(exp(x) * cos(exp(x)), x) == sin(exp(x))
    assert manualintegrate(exp(2 * x) * cos(exp(x)), x) == exp(x) * sin(exp(x)) + cos(exp(x))
    res = manualintegrate(exp(10 * x) * sin(exp(x)), x)
    assert not res.has(Integral)
    assert res.diff(x) == exp(10 * x) * sin(exp(x))
    assert manualintegrate(sum([x * exp(k * x) for k in range(1, 8)]), x) == x * exp(7 * x) / 7 + x * exp(6 * x) / 6 + x * exp(5 * x) / 5 + x * exp(4 * x) / 4 + x * exp(3 * x) / 3 + x * exp(2 * x) / 2 + x * exp(x) - exp(7 * x) / 49 - exp(6 * x) / 36 - exp(5 * x) / 25 - exp(4 * x) / 16 - exp(3 * x) / 9 - exp(2 * x) / 4 - exp(x)
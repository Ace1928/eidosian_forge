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
def test_issue_6746():
    y = Symbol('y')
    n = Symbol('n')
    assert manualintegrate(y ** x, x) == Piecewise((y ** x / log(y), Ne(log(y), 0)), (x, True))
    assert manualintegrate(y ** (n * x), x) == Piecewise((Piecewise((y ** (n * x) / log(y), Ne(log(y), 0)), (n * x, True)) / n, Ne(n, 0)), (x, True))
    assert manualintegrate(exp(n * x), x) == Piecewise((exp(n * x) / n, Ne(n, 0)), (x, True))
    y = Symbol('y', positive=True)
    assert manualintegrate((y + 1) ** x, x) == (y + 1) ** x / log(y + 1)
    y = Symbol('y', zero=True)
    assert manualintegrate((y + 1) ** x, x) == x
    y = Symbol('y')
    n = Symbol('n', nonzero=True)
    assert manualintegrate(y ** (n * x), x) == Piecewise((y ** (n * x) / log(y), Ne(log(y), 0)), (n * x, True)) / n
    y = Symbol('y', positive=True)
    assert manualintegrate((y + 1) ** (n * x), x) == (y + 1) ** (n * x) / (n * log(y + 1))
    a = Symbol('a', negative=True)
    b = Symbol('b')
    assert manualintegrate(1 / (a + b * x ** 2), x) == atan(x / sqrt(a / b)) / (b * sqrt(a / b))
    b = Symbol('b', negative=True)
    assert manualintegrate(1 / (a + b * x ** 2), x) == atan(x / (sqrt(-a) * sqrt(-1 / b))) / (b * sqrt(-a) * sqrt(-1 / b))
    assert manualintegrate(1 / ((x ** a + y ** b + 4) * sqrt(a * x ** 2 + 1)), x) == y ** (-b) * Integral(x ** (-a) / (y ** (-b) * sqrt(a * x ** 2 + 1) + x ** (-a) * sqrt(a * x ** 2 + 1) + 4 * x ** (-a) * y ** (-b) * sqrt(a * x ** 2 + 1)), x)
    assert manualintegrate(1 / ((x ** 2 + 4) * sqrt(4 * x ** 2 + 1)), x) == Integral(1 / ((x ** 2 + 4) * sqrt(4 * x ** 2 + 1)), x)
    assert manualintegrate(1 / (x - a ** x + x * b ** 2), x) == Integral(1 / (-a ** x + b ** 2 * x + x), x)
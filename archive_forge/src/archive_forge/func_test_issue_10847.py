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
def test_issue_10847():
    assert manualintegrate(x ** 2 / (x ** 2 - c), x) == c * atan(x / sqrt(-c)) / sqrt(-c) + x
    rc = Symbol('c', real=True)
    assert manualintegrate(x ** 2 / (x ** 2 - rc), x) == rc * Piecewise((atan(x / sqrt(-rc)) / sqrt(-rc), rc < 0), ((log(-sqrt(rc) + x) - log(sqrt(rc) + x)) / (2 * sqrt(rc)), True)) + x
    assert manualintegrate(sqrt(x - y) * log(z / x), x) == 4 * y ** Rational(3, 2) * atan(sqrt(x - y) / sqrt(y)) / 3 - 4 * y * sqrt(x - y) / 3 + 2 * (x - y) ** Rational(3, 2) * log(z / x) / 3 + 4 * (x - y) ** Rational(3, 2) / 9
    ry = Symbol('y', real=True)
    rz = Symbol('z', real=True)
    assert manualintegrate(sqrt(x - ry) * log(rz / x), x) == 4 * ry ** 2 * Piecewise((atan(sqrt(x - ry) / sqrt(ry)) / sqrt(ry), ry > 0), ((log(-sqrt(-ry) + sqrt(x - ry)) - log(sqrt(-ry) + sqrt(x - ry))) / (2 * sqrt(-ry)), True)) / 3 - 4 * ry * sqrt(x - ry) / 3 + 2 * (x - ry) ** Rational(3, 2) * log(rz / x) / 3 + 4 * (x - ry) ** Rational(3, 2) / 9
    assert manualintegrate(sqrt(x) * log(x), x) == 2 * x ** Rational(3, 2) * log(x) / 3 - 4 * x ** Rational(3, 2) / 9
    assert manualintegrate(sqrt(a * x + b) / x, x) == Piecewise((2 * b * atan(sqrt(a * x + b) / sqrt(-b)) / sqrt(-b) + 2 * sqrt(a * x + b), Ne(a, 0)), (sqrt(b) * log(x), True))
    ra = Symbol('a', real=True)
    rb = Symbol('b', real=True)
    assert manualintegrate(sqrt(ra * x + rb) / x, x) == Piecewise((-2 * rb * Piecewise((-atan(sqrt(ra * x + rb) / sqrt(-rb)) / sqrt(-rb), rb < 0), (-I * (log(-sqrt(rb) + sqrt(ra * x + rb)) - log(sqrt(rb) + sqrt(ra * x + rb))) / (2 * sqrt(-rb)), True)) + 2 * sqrt(ra * x + rb), Ne(ra, 0)), (sqrt(rb) * log(x), True))
    assert expand(manualintegrate(sqrt(ra * x + rb) / (x + rc), x)) == Piecewise((-2 * ra * rc * Piecewise((atan(sqrt(ra * x + rb) / sqrt(ra * rc - rb)) / sqrt(ra * rc - rb), ra * rc - rb > 0), (log(-sqrt(-ra * rc + rb) + sqrt(ra * x + rb)) / (2 * sqrt(-ra * rc + rb)) - log(sqrt(-ra * rc + rb) + sqrt(ra * x + rb)) / (2 * sqrt(-ra * rc + rb)), True)) + 2 * rb * Piecewise((atan(sqrt(ra * x + rb) / sqrt(ra * rc - rb)) / sqrt(ra * rc - rb), ra * rc - rb > 0), (log(-sqrt(-ra * rc + rb) + sqrt(ra * x + rb)) / (2 * sqrt(-ra * rc + rb)) - log(sqrt(-ra * rc + rb) + sqrt(ra * x + rb)) / (2 * sqrt(-ra * rc + rb)), True)) + 2 * sqrt(ra * x + rb), Ne(ra, 0)), (sqrt(rb) * log(rc + x), True))
    assert manualintegrate(sqrt(2 * x + 3) / (x + 1), x) == 2 * sqrt(2 * x + 3) - log(sqrt(2 * x + 3) + 1) + log(sqrt(2 * x + 3) - 1)
    assert manualintegrate(sqrt(2 * x + 3) / 2 * x, x) == (2 * x + 3) ** Rational(5, 2) / 20 - (2 * x + 3) ** Rational(3, 2) / 4
    assert manualintegrate(x ** Rational(3, 2) * log(x), x) == 2 * x ** Rational(5, 2) * log(x) / 5 - 4 * x ** Rational(5, 2) / 25
    assert manualintegrate(x ** (-3) * log(x), x) == -log(x) / (2 * x ** 2) - 1 / (4 * x ** 2)
    assert manualintegrate(log(y) / (y ** 2 * (1 - 1 / y)), y) == log(y) * log(-1 + 1 / y) - Integral(log(-1 + 1 / y) / y, y)
import math
from sympy.concrete.summations import (Sum, summation)
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function, Lambda, diff)
from sympy.core import EulerGamma
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (Abs, im, polar_lift, re, sign)
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (acosh, asinh, cosh, coth, csch, sinh, tanh, sech)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, sinc, tan, sec)
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.error_functions import (Ci, Ei, Si, erf, erfc, erfi, fresnelc, li)
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import lerchphi
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import And
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (Poly, factor)
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.sets.sets import Interval
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.tensor.indexed import (Idx, IndexedBase)
from sympy.core.expr import unchanged
from sympy.functions.elementary.integers import floor
from sympy.integrals.integrals import Integral
from sympy.integrals.risch import NonElementaryIntegral
from sympy.physics import units
from sympy.testing.pytest import (raises, slow, skip, ON_CI,
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.random import verify_numerically
def test_issue_21024():
    x = Symbol('x', real=True, nonzero=True)
    f = log(x) * log(4 * x) + log(3 * x + exp(2))
    F = x * log(x) ** 2 + x * (1 - 2 * log(2)) + (-2 * x + 2 * x * log(2)) * log(x) + (x + exp(2) / 6) * log(3 * x + exp(2)) + exp(2) * log(3 * x + exp(2)) / 6
    assert F == integrate(f, x)
    f = (x + exp(3)) / x ** 2
    F = log(x) - exp(3) / x
    assert F == integrate(f, x)
    f = (x ** 2 + exp(5)) / x
    F = x ** 2 / 2 + exp(5) * log(x)
    assert F == integrate(f, x)
    f = x / (2 * x + tanh(1))
    F = x / 2 - log(2 * x + tanh(1)) * tanh(1) / 4
    assert F == integrate(f, x)
    f = x - sinh(4) / x
    F = x ** 2 / 2 - log(x) * sinh(4)
    assert F == integrate(f, x)
    f = log(x + exp(5) / x)
    F = x * log(x + exp(5) / x) - x + 2 * exp(Rational(5, 2)) * atan(x * exp(Rational(-5, 2)))
    assert F == integrate(f, x)
    f = x ** 5 / (x + E)
    F = x ** 5 / 5 - E * x ** 4 / 4 + x ** 3 * exp(2) / 3 - x ** 2 * exp(3) / 2 + x * exp(4) - exp(5) * log(x + E)
    assert F == integrate(f, x)
    f = 4 * x / (x + sinh(5))
    F = 4 * x - 4 * log(x + sinh(5)) * sinh(5)
    assert F == integrate(f, x)
    f = x ** 2 / (2 * x + sinh(2))
    F = x ** 2 / 4 - x * sinh(2) / 4 + log(2 * x + sinh(2)) * sinh(2) ** 2 / 8
    assert F == integrate(f, x)
    f = -x ** 2 / (x + E)
    F = -x ** 2 / 2 + E * x - exp(2) * log(x + E)
    assert F == integrate(f, x)
    f = (2 * x + 3) * exp(5) / x
    F = 2 * x * exp(5) + 3 * exp(5) * log(x)
    assert F == integrate(f, x)
    f = x + 2 + cosh(3) / x
    F = x ** 2 / 2 + 2 * x + log(x) * cosh(3)
    assert F == integrate(f, x)
    f = x - tanh(1) / x ** 3
    F = x ** 2 / 2 + tanh(1) / (2 * x ** 2)
    assert F == integrate(f, x)
    f = (3 * x - exp(6)) / x
    F = 3 * x - exp(6) * log(x)
    assert F == integrate(f, x)
    f = x ** 4 / (x + exp(5)) ** 2 + x
    F = x ** 3 / 3 + x ** 2 * (Rational(1, 2) - exp(5)) + 3 * x * exp(10) - 4 * exp(15) * log(x + exp(5)) - exp(20) / (x + exp(5))
    assert F == integrate(f, x)
    f = x * (x + exp(10) / x ** 2) + x
    F = x ** 3 / 3 + x ** 2 / 2 + exp(10) * log(x)
    assert F == integrate(f, x)
    f = x + x / (5 * x + sinh(3))
    F = x ** 2 / 2 + x / 5 - log(5 * x + sinh(3)) * sinh(3) / 25
    assert F == integrate(f, x)
    f = (x + exp(3)) / (2 * x ** 2 + 2 * x)
    F = exp(3) * log(x) / 2 - exp(3) * log(x + 1) / 2 + log(x + 1) / 2
    assert F == integrate(f, x).expand()
    f = log(x + 4 * sinh(4))
    F = x * log(x + 4 * sinh(4)) - x + 4 * log(x + 4 * sinh(4)) * sinh(4)
    assert F == integrate(f, x)
    f = -x + 20 * (exp(-5) - atan(4) / x) ** 3 * sin(4) / x
    F = (-x ** 2 * exp(15) / 2 + 20 * log(x) * sin(4) - (-180 * x ** 2 * exp(5) * sin(4) * atan(4) + 90 * x * exp(10) * sin(4) * atan(4) ** 2 - 20 * exp(15) * sin(4) * atan(4) ** 3) / (3 * x ** 3)) * exp(-15)
    assert F == integrate(f, x)
    f = 2 * x ** 2 * exp(-4) + 6 / x
    F_true = (2 * x ** 3 / 3 + 6 * exp(4) * log(x)) * exp(-4)
    assert F_true == integrate(f, x)
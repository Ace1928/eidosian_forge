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
def test_issue_5547():
    L = Symbol('L')
    z = Symbol('z')
    r0 = Symbol('r0')
    R0 = Symbol('R0')
    assert integrate(r0 ** 2 * cos(z) ** 2, (z, -L / 2, L / 2)) == -r0 ** 2 * (-L / 4 - sin(L / 2) * cos(L / 2) / 2) + r0 ** 2 * (L / 4 + sin(L / 2) * cos(L / 2) / 2)
    assert integrate(r0 ** 2 * cos(R0 * z) ** 2, (z, -L / 2, L / 2)) == Piecewise((-r0 ** 2 * (-L * R0 / 4 - sin(L * R0 / 2) * cos(L * R0 / 2) / 2) / R0 + r0 ** 2 * (L * R0 / 4 + sin(L * R0 / 2) * cos(L * R0 / 2) / 2) / R0, (R0 > -oo) & (R0 < oo) & Ne(R0, 0)), (L * r0 ** 2, True))
    w = 2 * pi * z / L
    sol = sqrt(2) * sqrt(L) * r0 ** 2 * fresnelc(sqrt(2) * sqrt(L)) * gamma(S.One / 4) / (16 * gamma(S(5) / 4)) + L * r0 ** 2 / 2
    assert integrate(r0 ** 2 * cos(w * z) ** 2, (z, -L / 2, L / 2)) == sol
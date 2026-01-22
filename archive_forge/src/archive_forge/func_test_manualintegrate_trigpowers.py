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
def test_manualintegrate_trigpowers():
    assert manualintegrate(sin(x) ** 2 * cos(x), x) == sin(x) ** 3 / 3
    assert manualintegrate(sin(x) ** 2 * cos(x) ** 2, x) == x / 8 - sin(4 * x) / 32
    assert manualintegrate(sin(x) * cos(x) ** 3, x) == -cos(x) ** 4 / 4
    assert manualintegrate(sin(x) ** 3 * cos(x) ** 2, x) == cos(x) ** 5 / 5 - cos(x) ** 3 / 3
    assert manualintegrate(tan(x) ** 3 * sec(x), x) == sec(x) ** 3 / 3 - sec(x)
    assert manualintegrate(tan(x) * sec(x) ** 2, x) == sec(x) ** 2 / 2
    assert manualintegrate(cot(x) ** 5 * csc(x), x) == -csc(x) ** 5 / 5 + 2 * csc(x) ** 3 / 3 - csc(x)
    assert manualintegrate(cot(x) ** 2 * csc(x) ** 6, x) == -cot(x) ** 7 / 7 - 2 * cot(x) ** 5 / 5 - cot(x) ** 3 / 3
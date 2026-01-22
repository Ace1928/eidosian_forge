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
def test_manualintegrate_orthogonal_poly():
    n = symbols('n')
    a, b = (7, Rational(5, 3))
    polys = [jacobi(n, a, b, x), gegenbauer(n, a, x), chebyshevt(n, x), chebyshevu(n, x), legendre(n, x), hermite(n, x), laguerre(n, x), assoc_laguerre(n, a, x)]
    for p in polys:
        integral = manualintegrate(p, x)
        for deg in [-2, -1, 0, 1, 3, 5, 8]:
            try:
                p_subbed = p.subs(n, deg)
            except ValueError:
                continue
            assert (integral.subs(n, deg).diff(x) - p_subbed).expand() == 0
        q = x * p.subs(x, 2 * x + 1)
        integral = manualintegrate(q, x)
        for deg in [2, 4, 7]:
            assert (integral.subs(n, deg).diff(x) - q.subs(n, deg)).expand() == 0
        t = symbols('t')
        for i in range(len(p.args) - 1):
            new_args = list(p.args)
            new_args[i] = t
            assert isinstance(manualintegrate(p.func(*new_args), t), Integral)
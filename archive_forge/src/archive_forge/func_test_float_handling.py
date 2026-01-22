from sympy.assumptions.ask import (Q, ask)
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core import (GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (E, Float, I, Rational, oo, pi)
from sympy.core.relational import (Eq, Gt, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.complexes import (Abs, arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (atanh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (cbrt, root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, atan2, cos, sec, sin, tan)
from sympy.functions.special.error_functions import (erf, erfc, erfcinv, erfinv)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import Matrix
from sympy.matrices import SparseMatrix
from sympy.polys.polytools import Poly
from sympy.printing.str import sstr
from sympy.simplify.radsimp import denom
from sympy.solvers.solvers import (nsolve, solve, solve_linear)
from sympy.core.function import nfloat
from sympy.solvers import solve_linear_system, solve_linear_system_LU, \
from sympy.solvers.bivariate import _filtered_gens, _solve_lambert, _lambert
from sympy.solvers.solvers import _invert, unrad, checksol, posify, _ispow, \
from sympy.physics.units import cm
from sympy.polys.rootoftools import CRootOf
from sympy.testing.pytest import slow, XFAIL, SKIP, raises
from sympy.core.random import verify_numerically as tn
from sympy.abc import a, b, c, d, e, k, h, p, x, y, z, t, q, m, R
def test_float_handling():

    def test(e1, e2):
        return len(e1.atoms(Float)) == len(e2.atoms(Float))
    assert solve(x - 0.5, rational=True)[0].is_Rational
    assert solve(x - 0.5, rational=False)[0].is_Float
    assert solve(x - S.Half, rational=False)[0].is_Rational
    assert solve(x - 0.5, rational=None)[0].is_Float
    assert solve(x - S.Half, rational=None)[0].is_Rational
    assert test(nfloat(1 + 2 * x), 1.0 + 2.0 * x)
    for contain in [list, tuple, set]:
        ans = nfloat(contain([1 + 2 * x]))
        assert type(ans) is contain and test(list(ans)[0], 1.0 + 2.0 * x)
    k, v = list(nfloat({2 * x: [1 + 2 * x]}).items())[0]
    assert test(k, 2 * x) and test(v[0], 1.0 + 2.0 * x)
    assert test(nfloat(cos(2 * x)), cos(2.0 * x))
    assert test(nfloat(3 * x ** 2), 3.0 * x ** 2)
    assert test(nfloat(3 * x ** 2, exponent=True), 3.0 * x ** 2.0)
    assert test(nfloat(exp(2 * x)), exp(2.0 * x))
    assert test(nfloat(x / 3), x / 3.0)
    assert test(nfloat(x ** 4 + 2 * x + cos(Rational(1, 3)) + 1), x ** 4 + 2.0 * x + 1.94495694631474)
    tot = 100 + c + z + t
    assert solve(((0.7 + c) / tot - 0.6, (0.2 + z) / tot - 0.3, t / tot - 0.1)) == []
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import unchanged
from sympy.core.function import (Function, diff, expand)
from sympy.core.mul import Mul
from sympy.core.mod import Mod
from sympy.core.numbers import (Float, I, Rational, oo, pi, zoo)
from sympy.core.relational import (Eq, Ge, Gt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (Abs, adjoint, arg, conjugate, im, re, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import (Piecewise,
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, ITE, Not, Or)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.printing import srepr
from sympy.sets.contains import Contains
from sympy.sets.sets import Interval
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises, slow
from sympy.utilities.lambdify import lambdify
def test_piecewise_exclusive():
    p = Piecewise((0, x < 0), (S.Half, x <= 0), (1, True))
    assert piecewise_exclusive(p) == Piecewise((0, x < 0), (S.Half, Eq(x, 0)), (1, x > 0), evaluate=False)
    assert piecewise_exclusive(p + 2) == Piecewise((0, x < 0), (S.Half, Eq(x, 0)), (1, x > 0), evaluate=False) + 2
    assert piecewise_exclusive(Piecewise((1, y <= 0), (-Piecewise((2, y >= 0)), True))) == Piecewise((1, y <= 0), (-Piecewise((2, y >= 0), (S.NaN, y < 0), evaluate=False), y > 0), evaluate=False)
    assert piecewise_exclusive(Piecewise((1, x > y))) == Piecewise((1, x > y), (S.NaN, x <= y), evaluate=False)
    assert piecewise_exclusive(Piecewise((1, x > y)), skip_nan=True) == Piecewise((1, x > y))
    xr, yr = symbols('xr, yr', real=True)
    p1 = Piecewise((1, xr < 0), (2, True), evaluate=False)
    p1x = Piecewise((1, xr < 0), (2, xr >= 0), evaluate=False)
    p2 = Piecewise((p1, yr < 0), (3, True), evaluate=False)
    p2x = Piecewise((p1, yr < 0), (3, yr >= 0), evaluate=False)
    p2xx = Piecewise((p1x, yr < 0), (3, yr >= 0), evaluate=False)
    assert piecewise_exclusive(p2) == p2xx
    assert piecewise_exclusive(p2, deep=False) == p2x
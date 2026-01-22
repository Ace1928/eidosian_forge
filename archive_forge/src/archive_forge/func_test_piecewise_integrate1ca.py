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
@slow
def test_piecewise_integrate1ca():
    y = symbols('y', real=True)
    g = Piecewise((1 - x, Interval(0, 1).contains(x)), (1 + x, Interval(-1, 0).contains(x)), (0, True))
    gy1 = g.integrate((x, y, 1))
    g1y = g.integrate((x, 1, y))
    assert g.integrate((x, -2, 1)) == gy1.subs(y, -2)
    assert g.integrate((x, 1, -2)) == g1y.subs(y, -2)
    assert g.integrate((x, 0, 1)) == gy1.subs(y, 0)
    assert g.integrate((x, 1, 0)) == g1y.subs(y, 0)
    assert g.integrate((x, 2, 1)) == gy1.subs(y, 2)
    assert g.integrate((x, 1, 2)) == g1y.subs(y, 2)
    assert piecewise_fold(gy1.rewrite(Piecewise)).simplify() == Piecewise((1, y <= -1), (-y ** 2 / 2 - y + S.Half, y <= 0), (y ** 2 / 2 - y + S.Half, y < 1), (0, True))
    assert piecewise_fold(g1y.rewrite(Piecewise)).simplify() == Piecewise((-1, y <= -1), (y ** 2 / 2 + y - S.Half, y <= 0), (-y ** 2 / 2 + y - S.Half, y < 1), (0, True))
    assert gy1 == Piecewise((-Min(1, Max(-1, y)) ** 2 / 2 - Min(1, Max(-1, y)) + Min(1, Max(0, y)) ** 2 + S.Half, y < 1), (0, True))
    assert g1y == Piecewise((Min(1, Max(-1, y)) ** 2 / 2 + Min(1, Max(-1, y)) - Min(1, Max(0, y)) ** 2 - S.Half, y < 1), (0, True))
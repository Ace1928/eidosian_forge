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
def test_issue_11045():
    assert integrate(1 / (x * sqrt(x ** 2 - 1)), (x, 1, 2)) == pi / 3
    assert Piecewise((1, And(Or(x < 1, x > 3), x < 2)), (0, True)).integrate((x, 0, 3)) == 1
    assert Piecewise((1, x > 1), (2, x > x + 1), (3, True)).integrate((x, 0, 3)) == 5
    assert Piecewise((1, x > 1), (2, Eq(1, x)), (3, True)).integrate((x, 0, 4)) == 6
    assert Piecewise((1, And(2 * x > x + 1, x < 2)), (0, True)).integrate((x, 0, 3)) == 1
    assert Piecewise((1, Or(2 * x > x + 2, x < 1)), (0, True)).integrate((x, 0, 3)) == 2
    assert Piecewise((1, x > 1), (2, x > x + 1), (3, True)).integrate((x, 0, 3)) == 5
    assert Piecewise((2, Eq(1 - x, x * (1 / x - 1))), (0, True)).integrate((x, 0, 3)) == 6
    assert Piecewise((1, Or(x < 1, x > 2)), (2, x > 3), (3, True)).integrate((x, 0, 4)) == 6
    assert Piecewise((1, Ne(x, 0)), (2, True)).integrate((x, -1, 1)) == 2
    assert Piecewise((x, (x > 1) & (x < 3)), (1, x < 4)).integrate((x, 1, 4)) == 5
    p = Piecewise((x, (x > 1) & (x < 3)), (1, x < 4))
    nan = Undefined
    i = p.integrate((x, 1, y))
    assert i == Piecewise((y - 1, y < 1), (Min(3, y) ** 2 / 2 - Min(3, y) + Min(4, y) - S.Half, y <= Min(4, y)), (nan, True))
    assert p.integrate((x, 1, -1)) == i.subs(y, -1)
    assert p.integrate((x, 1, 4)) == 5
    assert p.integrate((x, 1, 5)) is nan
    p = Piecewise((1, x > 1), (2, Not(And(x > 1, x < 3))), (3, True))
    assert p.integrate((x, 0, 3)) == 4
    p = Piecewise((1, And(5 > x, x > 1)), (2, Or(x < 3, x > 7)), (4, x < 8))
    assert p.integrate((x, 0, 10)) == 20
    assert Piecewise((1, x < 1), (2, And(Eq(x, 3), x > 1))).integrate((x, 0, 3)) is S.NaN
    assert Piecewise((1, x < 1), (2, And(Eq(x, 3), x > 1)), (3, True)).integrate((x, 0, 3)) == 7
    assert Piecewise((1, x < 0), (2, And(Eq(x, 3), x < 1)), (3, True)).integrate((x, -1, 1)) == 4
    assert Piecewise((1, x < 1), (2, Eq(x, 3) & (y < x)), (3, True)).integrate((x, 0, 3)) == 7
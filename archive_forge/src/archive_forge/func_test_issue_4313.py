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
def test_issue_4313():
    u = Piecewise((0, x <= 0), (1, x >= a), (x / a, True))
    e = (u - u.subs(x, y)) ** 2 / (x - y) ** 2
    M = Max(0, a)
    assert integrate(e, x).expand() == Piecewise((Piecewise((0, x <= 0), (-y ** 2 / (a ** 2 * x - a ** 2 * y) + x / a ** 2 - 2 * y * log(-y) / a ** 2 + 2 * y * log(x - y) / a ** 2 - y / a ** 2, x <= M), (-y ** 2 / (-a ** 2 * y + a ** 2 * M) + 1 / (-y + M) - 1 / (x - y) - 2 * y * log(-y) / a ** 2 + 2 * y * log(-y + M) / a ** 2 - y / a ** 2 + M / a ** 2, True)), (a <= y) & (y <= 0) | (y <= 0) & (y > -oo)), (Piecewise((-1 / (x - y), x <= 0), (-a ** 2 / (a ** 2 * x - a ** 2 * y) + 2 * a * y / (a ** 2 * x - a ** 2 * y) - y ** 2 / (a ** 2 * x - a ** 2 * y) + 2 * log(-y) / a - 2 * log(x - y) / a + 2 / a + x / a ** 2 - 2 * y * log(-y) / a ** 2 + 2 * y * log(x - y) / a ** 2 - y / a ** 2, x <= M), (-a ** 2 / (-a ** 2 * y + a ** 2 * M) + 2 * a * y / (-a ** 2 * y + a ** 2 * M) - y ** 2 / (-a ** 2 * y + a ** 2 * M) + 2 * log(-y) / a - 2 * log(-y + M) / a + 2 / a - 2 * y * log(-y) / a ** 2 + 2 * y * log(-y + M) / a ** 2 - y / a ** 2 + M / a ** 2, True)), a <= y), (Piecewise((-y ** 2 / (a ** 2 * x - a ** 2 * y), x <= 0), (x / a ** 2 + y / a ** 2, x <= M), (a ** 2 / (-a ** 2 * y + a ** 2 * M) - a ** 2 / (a ** 2 * x - a ** 2 * y) - 2 * a * y / (-a ** 2 * y + a ** 2 * M) + 2 * a * y / (a ** 2 * x - a ** 2 * y) + y ** 2 / (-a ** 2 * y + a ** 2 * M) - y ** 2 / (a ** 2 * x - a ** 2 * y) + y / a ** 2 + M / a ** 2, True)), True))
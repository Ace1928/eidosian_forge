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
def test_issue_8919():
    c = symbols('c:5')
    x = symbols('x')
    f1 = Piecewise((c[1], x < 1), (c[2], True))
    f2 = Piecewise((c[3], x < Rational(1, 3)), (c[4], True))
    assert integrate(f1 * f2, (x, 0, 2)) == c[1] * c[3] / 3 + 2 * c[1] * c[4] / 3 + c[2] * c[4]
    f1 = Piecewise((0, x < 1), (2, True))
    f2 = Piecewise((3, x < 2), (0, True))
    assert integrate(f1 * f2, (x, 0, 3)) == 6
    y = symbols('y', positive=True)
    a, b, c, x, z = symbols('a,b,c,x,z', real=True)
    I = Integral(Piecewise((0, (x >= y) | (x < 0) | (b > c)), (a, True)), (x, 0, z))
    ans = I.doit()
    assert ans == Piecewise((0, b > c), (a * Min(y, z) - a * Min(0, z), True))
    for cond in (True, False):
        for yy in range(1, 3):
            for zz in range(-yy, 0, yy):
                reps = [(b > c, cond), (y, yy), (z, zz)]
                assert ans.subs(reps) == I.subs(reps).doit()
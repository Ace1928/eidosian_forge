from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import unchanged
from sympy.core.function import (count_ops, diff, expand, expand_multinomial, Function, Derivative)
from sympy.core.mul import Mul, _keep_coeff
from sympy.core import GoldenRatio
from sympy.core.numbers import (E, Float, I, oo, pi, Rational, zoo)
from sympy.core.relational import (Eq, Lt, Gt, Ge, Le)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, sign)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, csch, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, sinc, tan)
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.geometry.polygon import rad
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import (Matrix, eye)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys.polytools import (factor, Poly)
from sympy.simplify.simplify import (besselsimp, hypersimp, inversecombine, logcombine, nsimplify, nthroot, posify, separatevars, signsimp, simplify)
from sympy.solvers.solvers import solve
from sympy.testing.pytest import XFAIL, slow, _both_exp_pow
from sympy.abc import x, y, z, t, a, b, c, d, e, f, g, h, i, n
def test_logcombine_1():
    x, y = symbols('x,y')
    a = Symbol('a')
    z, w = symbols('z,w', positive=True)
    b = Symbol('b', real=True)
    assert logcombine(log(x) + 2 * log(y)) == log(x) + 2 * log(y)
    assert logcombine(log(x) + 2 * log(y), force=True) == log(x * y ** 2)
    assert logcombine(a * log(w) + log(z)) == a * log(w) + log(z)
    assert logcombine(b * log(z) + b * log(x)) == log(z ** b) + b * log(x)
    assert logcombine(b * log(z) - log(w)) == log(z ** b / w)
    assert logcombine(log(x) * log(z)) == log(x) * log(z)
    assert logcombine(log(w) * log(x)) == log(w) * log(x)
    assert logcombine(cos(-2 * log(z) + b * log(w))) in [cos(log(w ** b / z ** 2)), cos(log(z ** 2 / w ** b))]
    assert logcombine(log(log(x) - log(y)) - log(z), force=True) == log(log(x / y) / z)
    assert logcombine((2 + I) * log(x), force=True) == (2 + I) * log(x)
    assert logcombine((x ** 2 + log(x) - log(y)) / (x * y), force=True) == (x ** 2 + log(x / y)) / (x * y)
    assert logcombine(log(x) * 2 * log(y) + log(z), force=True) == log(z * y ** log(x ** 2))
    assert logcombine((x * y + sqrt(x ** 4 + y ** 4) + log(x) - log(y)) / (pi * x ** Rational(2, 3) * sqrt(y) ** 3), force=True) == (x * y + sqrt(x ** 4 + y ** 4) + log(x / y)) / (pi * x ** Rational(2, 3) * y ** Rational(3, 2))
    assert logcombine(gamma(-log(x / y)) * acos(-log(x / y)), force=True) == acos(-log(x / y)) * gamma(-log(x / y))
    assert logcombine(2 * log(z) * log(w) * log(x) + log(z) + log(w)) == log(z ** log(w ** 2)) * log(x) + log(w * z)
    assert logcombine(3 * log(w) + 3 * log(z)) == log(w ** 3 * z ** 3)
    assert logcombine(x * (y + 1) + log(2) + log(3)) == x * (y + 1) + log(6)
    assert logcombine((x + y) * log(w) + (-x - y) * log(3)) == (x + y) * log(w / 3)
    assert logcombine(log(x) + log(2)) == log(2 * x)
    eq = log(abs(x)) + log(abs(y))
    assert logcombine(eq) == eq
    reps = {x: 0, y: 0}
    assert log(abs(x) * abs(y)).subs(reps) != eq.subs(reps)
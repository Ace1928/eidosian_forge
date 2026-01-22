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
def test_nsimplify():
    x = Symbol('x')
    assert nsimplify(0) == 0
    assert nsimplify(-1) == -1
    assert nsimplify(1) == 1
    assert nsimplify(1 + x) == 1 + x
    assert nsimplify(2.7) == Rational(27, 10)
    assert nsimplify(1 - GoldenRatio) == (1 - sqrt(5)) / 2
    assert nsimplify((1 + sqrt(5)) / 4, [GoldenRatio]) == GoldenRatio / 2
    assert nsimplify(2 / GoldenRatio, [GoldenRatio]) == 2 * GoldenRatio - 2
    assert nsimplify(exp(pi * I * Rational(5, 3), evaluate=False)) == sympify('1/2 - sqrt(3)*I/2')
    assert nsimplify(sin(pi * Rational(3, 5), evaluate=False)) == sympify('sqrt(sqrt(5)/8 + 5/8)')
    assert nsimplify(sqrt(atan('1', evaluate=False)) * (2 + I), [pi]) == sqrt(pi) + sqrt(pi) / 2 * I
    assert nsimplify(2 + exp(2 * atan('1/4') * I)) == sympify('49/17 + 8*I/17')
    assert nsimplify(pi, tolerance=0.01) == Rational(22, 7)
    assert nsimplify(pi, tolerance=0.001) == Rational(355, 113)
    assert nsimplify(0.33333, tolerance=0.0001) == Rational(1, 3)
    assert nsimplify(2.0 ** (1 / 3.0), tolerance=0.001) == Rational(635, 504)
    assert nsimplify(2.0 ** (1 / 3.0), tolerance=0.001, full=True) == 2 ** Rational(1, 3)
    assert nsimplify(x + 0.5, rational=True) == S.Half + x
    assert nsimplify(1 / 0.3 + x, rational=True) == Rational(10, 3) + x
    assert nsimplify(log(3).n(), rational=True) == sympify('109861228866811/100000000000000')
    assert nsimplify(Float(0.27219826128795), [pi, log(2)]) == pi * log(2) / 8
    assert nsimplify(Float(0.27219826128795).n(3), [pi, log(2)]) == -pi / 4 - log(2) + Rational(7, 4)
    assert nsimplify(x / 7.0) == x / 7
    assert nsimplify(pi / 100.0) == pi / 100
    assert nsimplify(pi / 100.0, rational=False) == pi / 100.0
    assert nsimplify(pi / 1e-07) == 10000000 * pi
    assert not nsimplify(factor(-3.0 * z ** 2 * (z ** 2) ** (-2.5) + 3 * (z ** 2) ** (-1.5))).atoms(Float)
    e = x ** 0.0
    assert e.is_Pow and nsimplify(x ** 0.0) == 1
    assert nsimplify(3.333333, tolerance=0.1, rational=True) == Rational(10, 3)
    assert nsimplify(3.333333, tolerance=0.01, rational=True) == Rational(10, 3)
    assert nsimplify(3.666666, tolerance=0.1, rational=True) == Rational(11, 3)
    assert nsimplify(3.666666, tolerance=0.01, rational=True) == Rational(11, 3)
    assert nsimplify(33, tolerance=10, rational=True) == Rational(33)
    assert nsimplify(33.33, tolerance=10, rational=True) == Rational(30)
    assert nsimplify(37.76, tolerance=10, rational=True) == Rational(40)
    assert nsimplify(-203.1) == Rational(-2031, 10)
    assert nsimplify(0.2, tolerance=0) == Rational(1, 5)
    assert nsimplify(-0.2, tolerance=0) == Rational(-1, 5)
    assert nsimplify(0.2222, tolerance=0) == Rational(1111, 5000)
    assert nsimplify(-0.2222, tolerance=0) == Rational(-1111, 5000)
    assert nsimplify(S(2e-08)) == Rational(1, 50000000)
    assert nsimplify(1e-42, rational=True) != 0
    inf = Float('inf')
    infs = (-oo, oo, inf, -inf)
    for zi in infs:
        ans = sign(zi) * oo
        assert nsimplify(zi) == ans
        assert nsimplify(zi + x) == x + ans
    assert nsimplify(0.33333333, rational=True, rational_conversion='exact') == Rational(0.33333333)
    assert nsimplify(pi.evalf(100) * x, rational_conversion='exact').evalf(100) == pi.evalf(100) * x
from sympy.assumptions.refine import refine
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import expand_log
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (adjoint, conjugate, re, sign, transpose)
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys.polytools import gcd
from sympy.series.order import O
from sympy.simplify.simplify import simplify
from sympy.core.parameters import global_parameters
from sympy.functions.elementary.exponential import match_real_imag
from sympy.abc import x, y, z
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises, XFAIL, _both_exp_pow
@_both_exp_pow
def test_exp_subs():
    x = Symbol('x')
    e = exp(3 * log(x), evaluate=False)
    assert e.subs(x ** 3, y ** 3) == e
    assert e.subs(x ** 2, 5) == e
    assert (x ** 3).subs(x ** 2, y) != y ** Rational(3, 2)
    assert exp(exp(x) + exp(x ** 2)).subs(exp(exp(x)), y) == y * exp(exp(x ** 2))
    assert exp(x).subs(E, y) == y ** x
    x = symbols('x', real=True)
    assert exp(5 * x).subs(exp(7 * x), y) == y ** Rational(5, 7)
    assert exp(2 * x + 7).subs(exp(3 * x), y) == y ** Rational(2, 3) * exp(7)
    x = symbols('x', positive=True)
    assert exp(3 * log(x)).subs(x ** 2, y) == y ** Rational(3, 2)
    assert exp(exp(x + E)).subs(exp, 3) == 3 ** 3 ** (x + E)
    assert exp(exp(x + E)).subs(exp, sin) == sin(sin(x + E))
    assert exp(exp(x + E)).subs(E, 3) == 3 ** 3 ** (x + 3)
    assert exp(3).subs(E, sin) == sin(3)
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
def test_log_leading_term():
    p = Symbol('p')
    assert log(1 + x + x ** 2).as_leading_term(x, cdir=1) == x
    assert log(2 * x).as_leading_term(x, cdir=1) == log(x) + log(2)
    assert log(2 * x).as_leading_term(x, cdir=-1) == log(x) + log(2)
    assert log(-2 * x).as_leading_term(x, cdir=1, logx=p) == p + log(2) + I * pi
    assert log(-2 * x).as_leading_term(x, cdir=-1, logx=p) == p + log(2) - I * pi
    assert log(-2 * x + (3 - I) * x ** 2).as_leading_term(x, cdir=1) == log(x) + log(2) - I * pi
    assert log(-2 * x + (3 - I) * x ** 2).as_leading_term(x, cdir=-1) == log(x) + log(2) - I * pi
    assert log(2 * x + (3 - I) * x ** 2).as_leading_term(x, cdir=1) == log(x) + log(2)
    assert log(2 * x + (3 - I) * x ** 2).as_leading_term(x, cdir=-1) == log(x) + log(2) - 2 * I * pi
    assert log(-1 + x - I * x ** 2 + I * x ** 3).as_leading_term(x, cdir=1) == -I * pi
    assert log(-1 + x - I * x ** 2 + I * x ** 3).as_leading_term(x, cdir=-1) == -I * pi
    assert log(-1 / (1 - x)).as_leading_term(x, cdir=1) == I * pi
    assert log(-1 / (1 - x)).as_leading_term(x, cdir=-1) == I * pi
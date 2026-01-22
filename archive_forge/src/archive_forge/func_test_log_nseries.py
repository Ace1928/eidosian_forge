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
def test_log_nseries():
    p = Symbol('p')
    assert log(1 / x)._eval_nseries(x, 4, logx=-p, cdir=1) == p
    assert log(1 / x)._eval_nseries(x, 4, logx=-p, cdir=-1) == p + 2 * I * pi
    assert log(x - 1)._eval_nseries(x, 4, None, I) == I * pi - x - x ** 2 / 2 - x ** 3 / 3 + O(x ** 4)
    assert log(x - 1)._eval_nseries(x, 4, None, -I) == -I * pi - x - x ** 2 / 2 - x ** 3 / 3 + O(x ** 4)
    assert log(I * x + I * x ** 3 - 1)._eval_nseries(x, 3, None, 1) == I * pi - I * x + x ** 2 / 2 + O(x ** 3)
    assert log(I * x + I * x ** 3 - 1)._eval_nseries(x, 3, None, -1) == -I * pi - I * x + x ** 2 / 2 + O(x ** 3)
    assert log(I * x ** 2 + I * x ** 3 - 1)._eval_nseries(x, 3, None, 1) == I * pi - I * x ** 2 + O(x ** 3)
    assert log(I * x ** 2 + I * x ** 3 - 1)._eval_nseries(x, 3, None, -1) == I * pi - I * x ** 2 + O(x ** 3)
    assert log(2 * x + (3 - I) * x ** 2)._eval_nseries(x, 3, None, 1) == log(2) + log(x) + x * (S(3) / 2 - I / 2) + x ** 2 * (-1 + 3 * I / 4) + O(x ** 3)
    assert log(2 * x + (3 - I) * x ** 2)._eval_nseries(x, 3, None, -1) == -2 * I * pi + log(2) + log(x) - x * (-S(3) / 2 + I / 2) + x ** 2 * (-1 + 3 * I / 4) + O(x ** 3)
    assert log(-2 * x + (3 - I) * x ** 2)._eval_nseries(x, 3, None, 1) == -I * pi + log(2) + log(x) + x * (-S(3) / 2 + I / 2) + x ** 2 * (-1 + 3 * I / 4) + O(x ** 3)
    assert log(-2 * x + (3 - I) * x ** 2)._eval_nseries(x, 3, None, -1) == -I * pi + log(2) + log(x) - x * (S(3) / 2 - I / 2) + x ** 2 * (-1 + 3 * I / 4) + O(x ** 3)
    assert log(sqrt(-I * x ** 2 - 3) * sqrt(-I * x ** 2 - 1) - 2)._eval_nseries(x, 3, None, 1) == -I * pi + log(sqrt(3) + 2) + I * x ** 2 * (-2 + 4 * sqrt(3) / 3) + O(x ** 3)
    assert log(-1 / (1 - x))._eval_nseries(x, 3, None, 1) == I * pi + x + x ** 2 / 2 + O(x ** 3)
    assert log(-1 / (1 - x))._eval_nseries(x, 3, None, -1) == I * pi + x + x ** 2 / 2 + O(x ** 3)
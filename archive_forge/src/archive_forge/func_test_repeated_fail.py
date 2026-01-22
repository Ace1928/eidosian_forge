import sys
import builtins
import types
from sympy.assumptions import Q
from sympy.core import Symbol, Function, Float, Rational, Integer, I, Mul, Pow, Eq, Lt, Le, Gt, Ge, Ne
from sympy.functions import exp, factorial, factorial2, sin, Min, Max
from sympy.logic import And
from sympy.series import Limit
from sympy.testing.pytest import raises, skip
from sympy.parsing.sympy_parser import (
def test_repeated_fail():
    inputs = ['1[1]', '.1e1[1]', '0x1[1]', '1.1j[1]', '1.1[1 + 1]', '0.1[[1]]', '0x1.1[1]']
    for text in inputs:
        raises(TypeError, lambda: parse_expr(text))
    inputs = ['0.1[', '0.1[1', '0.1[]']
    for text in inputs:
        raises((TokenError, SyntaxError), lambda: parse_expr(text))
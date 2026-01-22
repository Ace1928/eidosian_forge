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
def test_convert_equals_signs():
    transformations = standard_transformations + (convert_equals_signs,)
    x = Symbol('x')
    y = Symbol('y')
    assert parse_expr('1*2=x', transformations=transformations) == Eq(2, x)
    assert parse_expr('y = x', transformations=transformations) == Eq(y, x)
    assert parse_expr('(2*y = x) = False', transformations=transformations) == Eq(Eq(2 * y, x), False)
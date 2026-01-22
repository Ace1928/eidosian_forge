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
def test_match_parentheses_implicit_multiplication():
    transformations = standard_transformations + (implicit_multiplication,)
    raises(TokenError, lambda: parse_expr('(1,2),(3,4]', transformations=transformations))
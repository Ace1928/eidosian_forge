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
def test_split_symbols_numeric():
    transformations = standard_transformations + (implicit_multiplication_application,)
    n = Symbol('n')
    expr1 = parse_expr('2**n * 3**n')
    expr2 = parse_expr('2**n3**n', transformations=transformations)
    assert expr1 == expr2 == 2 ** n * 3 ** n
    expr1 = parse_expr('n12n34', transformations=transformations)
    assert expr1 == n * 12 * n * 34
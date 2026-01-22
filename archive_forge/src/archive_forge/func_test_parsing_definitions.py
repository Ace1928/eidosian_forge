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
def test_parsing_definitions():
    from sympy.abc import x
    assert len(_transformation) == 12
    assert _transformation[0] == lambda_notation
    assert _transformation[1] == auto_symbol
    assert _transformation[2] == repeated_decimals
    assert _transformation[3] == auto_number
    assert _transformation[4] == factorial_notation
    assert _transformation[5] == implicit_multiplication_application
    assert _transformation[6] == convert_xor
    assert _transformation[7] == implicit_application
    assert _transformation[8] == implicit_multiplication
    assert _transformation[9] == convert_equals_signs
    assert _transformation[10] == function_exponentiation
    assert _transformation[11] == rationalize
    assert T[:5] == T[0, 1, 2, 3, 4] == standard_transformations
    t = _transformation
    assert T[-1, 0] == (t[len(t) - 1], t[0])
    assert T[:5, 8] == standard_transformations + (t[8],)
    assert parse_expr('0.3x^2', transformations='all') == 3 * x ** 2 / 10
    assert parse_expr('sin 3x', transformations='implicit') == sin(3 * x)
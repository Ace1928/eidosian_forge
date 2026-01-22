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
def test_python3_features():
    if sys.version_info < (3, 8):
        skip('test_python3_features requires Python 3.8 or newer')
    assert parse_expr('123_456') == 123456
    assert parse_expr('1.2[3_4]') == parse_expr('1.2[34]') == Rational(611, 495)
    assert parse_expr('1.2[012_012]') == parse_expr('1.2[012012]') == Rational(400, 333)
    assert parse_expr('.[3_4]') == parse_expr('.[34]') == Rational(34, 99)
    assert parse_expr('.1[3_4]') == parse_expr('.1[34]') == Rational(133, 990)
    assert parse_expr('123_123.123_123[3_4]') == parse_expr('123123.123123[34]') == Rational(12189189189211, 99000000)
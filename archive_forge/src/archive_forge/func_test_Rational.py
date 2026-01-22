import contextlib
import itertools
import re
import typing
from enum import Enum
from typing import Callable
import sympy
from sympy import Add, Implies, sqrt
from sympy.core import Mul, Pow
from sympy.core import (S, pi, symbols, Function, Rational, Integer,
from sympy.functions import Piecewise, exp, sin, cos
from sympy.printing.smtlib import smtlib_code
from sympy.testing.pytest import raises, Failed
def test_Rational():
    with _check_warns([_W.WILL_NOT_ASSERT] * 4) as w:
        assert smtlib_code(Rational(3, 7), log_warn=w) == '(/ 3 7)'
        assert smtlib_code(Rational(18, 9), log_warn=w) == '2'
        assert smtlib_code(Rational(3, -7), log_warn=w) == '(/ -3 7)'
        assert smtlib_code(Rational(-3, -7), log_warn=w) == '(/ 3 7)'
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT] * 2) as w:
        assert smtlib_code(x + Rational(3, 7), auto_declare=False, log_warn=w) == '(+ (/ 3 7) x)'
        assert smtlib_code(Rational(3, 7) * x, log_warn=w) == '(declare-const x Real)\n(* (/ 3 7) x)'
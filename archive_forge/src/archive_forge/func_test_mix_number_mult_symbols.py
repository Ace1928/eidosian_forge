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
def test_mix_number_mult_symbols():
    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(1 / pi, known_constants={pi: 'MY_PI'}, log_warn=w) == '(pow MY_PI -1)'
    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code([Eq(pi, 3.14, evaluate=False), 1 / pi], known_constants={pi: 'MY_PI'}, log_warn=w) == '(assert (= MY_PI 3.14))\n(pow MY_PI -1)'
    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(Add(S.Zero, S.One, S.NegativeOne, S.Half, S.Exp1, S.Pi, S.GoldenRatio, evaluate=False), known_constants={S.Pi: 'p', S.GoldenRatio: 'g', S.Exp1: 'e'}, known_functions={Add: 'plus', exp: 'exp'}, precision=3, log_warn=w) == '(plus 0 1 -1 (/ 1 2) (exp 1) p g)'
    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(Add(S.Zero, S.One, S.NegativeOne, S.Half, S.Exp1, S.Pi, S.GoldenRatio, evaluate=False), known_constants={S.Pi: 'p'}, known_functions={Add: 'plus', exp: 'exp'}, precision=3, log_warn=w) == '(plus 0 1 -1 (/ 1 2) (exp 1) p 1.62)'
    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(Add(S.Zero, S.One, S.NegativeOne, S.Half, S.Exp1, S.Pi, S.GoldenRatio, evaluate=False), known_functions={Add: 'plus'}, precision=3, log_warn=w) == '(plus 0 1 -1 (/ 1 2) 2.72 3.14 1.62)'
    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(Add(S.Zero, S.One, S.NegativeOne, S.Half, S.Exp1, S.Pi, S.GoldenRatio, evaluate=False), known_constants={S.Exp1: 'e'}, known_functions={Add: 'plus'}, precision=3, log_warn=w) == '(plus 0 1 -1 (/ 1 2) e 3.14 1.62)'
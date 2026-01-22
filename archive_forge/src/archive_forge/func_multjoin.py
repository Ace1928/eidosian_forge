from __future__ import annotations
from typing import Any
from sympy.core import Mul, Pow, S, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from re import search
def multjoin(a, a_str):
    r = a_str[0]
    for i in range(1, len(a)):
        mulsym = '*' if a[i - 1].is_number else '.*'
        r = r + mulsym + a_str[i]
    return r
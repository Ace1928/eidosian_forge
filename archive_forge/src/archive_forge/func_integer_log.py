from __future__ import annotations
from typing import Callable
from math import log as _log, sqrt as _sqrt
from itertools import product
from .sympify import _sympify
from .cache import cacheit
from .singleton import S
from .expr import Expr
from .evalf import PrecisionExhausted
from .function import (expand_complex, expand_multinomial,
from .logic import fuzzy_bool, fuzzy_not, fuzzy_and, fuzzy_or
from .parameters import global_parameters
from .relational import is_gt, is_lt
from .kind import NumberKind, UndefinedKind
from sympy.external.gmpy import HAS_GMPY, gmpy
from sympy.utilities.iterables import sift
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.misc import as_int
from sympy.multipledispatch import Dispatcher
from mpmath.libmp import sqrtrem as mpmath_sqrtrem
from .add import Add
from .numbers import Integer
from .mul import Mul, _keep_coeff
from .symbol import Symbol, Dummy, symbols
def integer_log(y, x):
    """
    Returns ``(e, bool)`` where e is the largest nonnegative integer
    such that :math:`|y| \\geq |x^e|` and ``bool`` is True if $y = x^e$.

    Examples
    ========

    >>> from sympy import integer_log
    >>> integer_log(125, 5)
    (3, True)
    >>> integer_log(17, 9)
    (1, False)
    >>> integer_log(4, -2)
    (2, True)
    >>> integer_log(-125,-5)
    (3, True)

    See Also
    ========
    integer_nthroot
    sympy.ntheory.primetest.is_square
    sympy.ntheory.factor_.multiplicity
    sympy.ntheory.factor_.perfect_power
    """
    if x == 1:
        raise ValueError('x cannot take value as 1')
    if y == 0:
        raise ValueError('y cannot take value as 0')
    if x in (-2, 2):
        x = int(x)
        y = as_int(y)
        e = y.bit_length() - 1
        return (e, x ** e == y)
    if x < 0:
        n, b = integer_log(y if y > 0 else -y, -x)
        return (n, b and bool(n % 2 if y < 0 else not n % 2))
    x = as_int(x)
    y = as_int(y)
    r = e = 0
    while y >= x:
        d = x
        m = 1
        while y >= d:
            y, rem = divmod(y, d)
            r = r or rem
            e += m
            if y > d:
                d *= d
                m *= 2
    return (e, r == 0 and y == 1)
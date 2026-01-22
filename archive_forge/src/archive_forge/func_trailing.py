from collections import defaultdict
from functools import reduce
import random
import math
from sympy.core import sympify
from sympy.core.containers import Dict
from sympy.core.evalf import bitcount
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.numbers import igcd, ilcm, Rational, Integer
from sympy.core.power import integer_nthroot, Pow, integer_log
from sympy.core.singleton import S
from sympy.external.gmpy import SYMPY_INTS
from .primetest import isprime
from .generate import sieve, primerange, nextprime
from .digits import digits
from sympy.utilities.iterables import flatten
from sympy.utilities.misc import as_int, filldedent
from .ecm import _ecm_one_factor
def trailing(n):
    """Count the number of trailing zero digits in the binary
    representation of n, i.e. determine the largest power of 2
    that divides n.

    Examples
    ========

    >>> from sympy import trailing
    >>> trailing(128)
    7
    >>> trailing(63)
    0
    """
    n = abs(int(n))
    if not n:
        return 0
    low_byte = n & 255
    if low_byte:
        return small_trailing[low_byte]
    z = bitcount(n) - 1
    if isinstance(z, SYMPY_INTS):
        if n == 1 << z:
            return z
    if z < 300:
        t = 8
        n >>= 8
        while not n & 255:
            n >>= 8
            t += 8
        return t + small_trailing[n & 255]
    t = 0
    p = 8
    while not n & 1:
        while not n & (1 << p) - 1:
            n >>= p
            t += p
            p *= 2
        p //= 2
    return t
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
def perfect_power(n, candidates=None, big=True, factor=True):
    """
    Return ``(b, e)`` such that ``n`` == ``b**e`` if ``n`` is a unique
    perfect power with ``e > 1``, else ``False`` (e.g. 1 is not a
    perfect power). A ValueError is raised if ``n`` is not Rational.

    By default, the base is recursively decomposed and the exponents
    collected so the largest possible ``e`` is sought. If ``big=False``
    then the smallest possible ``e`` (thus prime) will be chosen.

    If ``factor=True`` then simultaneous factorization of ``n`` is
    attempted since finding a factor indicates the only possible root
    for ``n``. This is True by default since only a few small factors will
    be tested in the course of searching for the perfect power.

    The use of ``candidates`` is primarily for internal use; if provided,
    False will be returned if ``n`` cannot be written as a power with one
    of the candidates as an exponent and factoring (beyond testing for
    a factor of 2) will not be attempted.

    Examples
    ========

    >>> from sympy import perfect_power, Rational
    >>> perfect_power(16)
    (2, 4)
    >>> perfect_power(16, big=False)
    (4, 2)

    Negative numbers can only have odd perfect powers:

    >>> perfect_power(-4)
    False
    >>> perfect_power(-8)
    (-2, 3)

    Rationals are also recognized:

    >>> perfect_power(Rational(1, 2)**3)
    (1/2, 3)
    >>> perfect_power(Rational(-3, 2)**3)
    (-3/2, 3)

    Notes
    =====

    To know whether an integer is a perfect power of 2 use

        >>> is2pow = lambda n: bool(n and not n & (n - 1))
        >>> [(i, is2pow(i)) for i in range(5)]
        [(0, False), (1, True), (2, True), (3, False), (4, True)]

    It is not necessary to provide ``candidates``. When provided
    it will be assumed that they are ints. The first one that is
    larger than the computed maximum possible exponent will signal
    failure for the routine.

        >>> perfect_power(3**8, [9])
        False
        >>> perfect_power(3**8, [2, 4, 8])
        (3, 8)
        >>> perfect_power(3**8, [4, 8], big=False)
        (9, 4)

    See Also
    ========
    sympy.core.power.integer_nthroot
    sympy.ntheory.primetest.is_square
    """
    if isinstance(n, Rational) and (not n.is_Integer):
        p, q = n.as_numer_denom()
        if p is S.One:
            pp = perfect_power(q)
            if pp:
                pp = (n.func(1, pp[0]), pp[1])
        else:
            pp = perfect_power(p)
            if pp:
                num, e = pp
                pq = perfect_power(q, [e])
                if pq:
                    den, _ = pq
                    pp = (n.func(num, den), e)
        return pp
    n = as_int(n)
    if n < 0:
        pp = perfect_power(-n)
        if pp:
            b, e = pp
            if e % 2:
                return (-b, e)
        return False
    if n <= 3:
        return False
    logn = math.log(n, 2)
    max_possible = int(logn) + 2
    not_square = n % 10 in [2, 3, 7, 8]
    min_possible = 2 + not_square
    if not candidates:
        candidates = primerange(min_possible, max_possible)
    else:
        candidates = sorted([i for i in candidates if min_possible <= i < max_possible])
        if n % 2 == 0:
            e = trailing(n)
            candidates = [i for i in candidates if e % i == 0]
        if big:
            candidates = reversed(candidates)
        for e in candidates:
            r, ok = integer_nthroot(n, e)
            if ok:
                return (r, e)
        return False

    def _factors():
        rv = 2 + n % 2
        while True:
            yield rv
            rv = nextprime(rv)
    for fac, e in zip(_factors(), candidates):
        if factor and n % fac == 0:
            if fac == 2:
                e = trailing(n)
            else:
                e = multiplicity(fac, n)
            if e == 1:
                return False
            r, exact = integer_nthroot(n, e)
            if not exact:
                m = n // fac ** e
                rE = perfect_power(m, candidates=divisors(e, generator=True))
                if not rE:
                    return False
                else:
                    r, E = rE
                    r, e = (fac ** (e // E) * r, E)
            if not big:
                e0 = primefactors(e)
                if e0[0] != e:
                    r, e = (r ** (e // e0[0]), e0[0])
            return (r, e)
        if logn / e < 40:
            b = 2.0 ** (logn / e)
            if abs(int(b + 0.5) - b) > 0.01:
                continue
        r, exact = integer_nthroot(n, e)
        if exact:
            if big:
                m = perfect_power(r, big=big, factor=factor)
                if m:
                    r, e = (m[0], e * m[1])
            return (int(r), e)
    return False
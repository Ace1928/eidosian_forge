from string import whitespace, ascii_uppercase as uppercase, printable
from functools import reduce
import warnings
from itertools import cycle
from sympy.core import Symbol
from sympy.core.numbers import igcdex, mod_inverse, igcd, Rational
from sympy.core.random import _randrange, _randint
from sympy.matrices import Matrix
from sympy.ntheory import isprime, primitive_root, factorint
from sympy.ntheory import totient as _euler
from sympy.ntheory import reduced_totient as _carmichael
from sympy.ntheory.generate import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import FF
from sympy.polys.polytools import gcd, Poly
from sympy.utilities.misc import as_int, filldedent, translate
from sympy.utilities.iterables import uniq, multiset
def lfsr_autocorrelation(L, P, k):
    """
    This function computes the LFSR autocorrelation function.

    Parameters
    ==========

    L
        A periodic sequence of elements of `GF(2)`.
        L must have length larger than P.

    P
        The period of L.

    k : int
        An integer `k` (`0 < k < P`).

    Returns
    =======

    autocorrelation
        The k-th value of the autocorrelation of the LFSR L.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     lfsr_sequence, lfsr_autocorrelation)
    >>> from sympy.polys.domains import FF
    >>> F = FF(2)
    >>> fill = [F(1), F(1), F(0), F(1)]
    >>> key = [F(1), F(0), F(0), F(1)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_autocorrelation(s, 15, 7)
    -1/15
    >>> lfsr_autocorrelation(s, 15, 0)
    1

    """
    if not isinstance(L, list):
        raise TypeError('L (=%s) must be a list' % L)
    P = int(P)
    k = int(k)
    L0 = L[:P]
    L1 = L0 + L0[:k]
    L2 = [(-1) ** (L1[i].to_int() + L1[i + k].to_int()) for i in range(P)]
    tot = sum(L2)
    return Rational(tot, P)
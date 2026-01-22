from functools import reduce
from math import prod
from sympy.core.numbers import igcdex, igcd
from sympy.ntheory.primetest import isprime
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt, gf_crt1, gf_crt2
from sympy.utilities.misc import as_int
def crt2(m, v, mm, e, s, symmetric=False):
    """Second part of Chinese Remainder Theorem, for multiple application.

    Examples
    ========

    >>> from sympy.ntheory.modular import crt1, crt2
    >>> mm, e, s = crt1([18, 42, 6])
    >>> crt2([18, 42, 6], [0, 0, 0], mm, e, s)
    (0, 4536)
    """
    result = gf_crt2(v, m, mm, e, s, ZZ)
    if symmetric:
        return (symmetric_residue(result, mm), mm)
    return (result, mm)
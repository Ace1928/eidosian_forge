from __future__ import annotations
from sympy.core.function import Function
from sympy.core.numbers import igcd, igcdex, mod_inverse
from sympy.core.power import isqrt
from sympy.core.singleton import S
from sympy.polys import Poly
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt1, gf_crt2, linear_congruence
from .primetest import isprime
from .factor_ import factorint, trailing, totient, multiplicity, perfect_power
from sympy.utilities.misc import as_int
from sympy.core.random import _randint, randint
from itertools import cycle, product
def nthroot_mod(a, n, p, all_roots=False):
    """
    Find the solutions to ``x**n = a mod p``.

    Parameters
    ==========

    a : integer
    n : positive integer
    p : positive integer
    all_roots : if False returns the smallest root, else the list of roots

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import nthroot_mod
    >>> nthroot_mod(11, 4, 19)
    8
    >>> nthroot_mod(11, 4, 19, True)
    [8, 11]
    >>> nthroot_mod(68, 3, 109)
    23
    """
    a = a % p
    a, n, p = (as_int(a), as_int(n), as_int(p))
    if n == 2:
        return sqrt_mod(a, p, all_roots)
    if not isprime(p):
        return _nthroot_mod_composite(a, n, p)
    if a % p == 0:
        return [0]
    if not is_nthpow_residue(a, n, p):
        return [] if all_roots else None
    if (p - 1) % n == 0:
        return _nthroot_mod1(a, n, p, all_roots)
    pa = n
    pb = p - 1
    b = 1
    if pa < pb:
        a, pa, b, pb = (b, pb, a, pa)
    while pb:
        q, r = divmod(pa, pb)
        c = pow(b, q, p)
        c = igcdex(c, p)[0]
        c = c * a % p
        pa, pb = (pb, r)
        a, b = (b, c)
    if pa == 1:
        if all_roots:
            res = [a]
        else:
            res = a
    elif pa == 2:
        return sqrt_mod(a, p, all_roots)
    else:
        res = _nthroot_mod1(a, pa, p, all_roots)
    return res
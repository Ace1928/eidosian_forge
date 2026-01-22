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
def quadratic_congruence(a, b, c, p):
    """
    Find the solutions to ``a x**2 + b x + c = 0 mod p.

    Parameters
    ==========

    a : int
    b : int
    c : int
    p : int
        A positive integer.
    """
    a = as_int(a)
    b = as_int(b)
    c = as_int(c)
    p = as_int(p)
    a = a % p
    b = b % p
    c = c % p
    if a == 0:
        return linear_congruence(b, -c, p)
    if p == 2:
        roots = []
        if c % 2 == 0:
            roots.append(0)
        if (a + b + c) % 2 == 0:
            roots.append(1)
        return roots
    if isprime(p):
        inv_a = mod_inverse(a, p)
        b *= inv_a
        c *= inv_a
        if b % 2 == 1:
            b = b + p
        d = (b * b // 4 - c) % p
        y = sqrt_mod(d, p, all_roots=True)
        res = set()
        for i in y:
            res.add((i - b // 2) % p)
        return sorted(res)
    y = sqrt_mod(b * b - 4 * a * c, 4 * a * p, all_roots=True)
    res = set()
    for i in y:
        root = linear_congruence(2 * a, i - b, 4 * a * p)
        for j in root:
            res.add(j % p)
    return sorted(res)
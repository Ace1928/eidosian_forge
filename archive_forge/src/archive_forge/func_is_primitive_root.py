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
def is_primitive_root(a, p):
    """
    Returns True if ``a`` is a primitive root of ``p``.

    ``a`` is said to be the primitive root of ``p`` if gcd(a, p) == 1 and
    totient(p) is the smallest positive number s.t.

        a**totient(p) cong 1 mod(p)

    Parameters
    ==========

    a : integer
    p : integer, p > 1. a and p should be relatively prime

    Examples
    ========

    >>> from sympy.ntheory import is_primitive_root, n_order, totient
    >>> is_primitive_root(3, 10)
    True
    >>> is_primitive_root(9, 10)
    False
    >>> n_order(3, 10) == totient(10)
    True
    >>> n_order(9, 10) == totient(10)
    False

    """
    a, p = (as_int(a), as_int(p))
    if p <= 1:
        raise ValueError('p should be an integer greater than 1')
    a = a % p
    if igcd(a, p) != 1:
        raise ValueError('The two numbers should be relatively prime')
    if p <= 4:
        return a == p - 1
    t = trailing(p)
    if t > 1:
        return False
    q = p >> t
    if isprime(q):
        group_order = q - 1
        factors = set(factorint(q - 1).keys())
    else:
        m = perfect_power(q)
        if not m:
            return False
        q, e = m
        if not isprime(q):
            return False
        group_order = q ** (e - 1) * (q - 1)
        factors = set(factorint(q - 1).keys())
        factors.add(q)
    return all((pow(a, group_order // prime, p) != 1 for prime in factors))
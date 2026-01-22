from sympy.core.add import Add
from sympy.core.numbers import AlgebraicNumber
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.ntheory import sieve
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import QQ
from sympy.polys.numberfields.minpoly import _choose_factor, minimal_polynomial
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly, PurePoly, factor_list
from sympy.utilities import public
from mpmath import MPContext
def is_isomorphism_possible(a, b):
    """Necessary but not sufficient test for isomorphism. """
    n = a.minpoly.degree()
    m = b.minpoly.degree()
    if m % n != 0:
        return False
    if n == m:
        return True
    da = a.minpoly.discriminant()
    db = b.minpoly.discriminant()
    i, k, half = (1, m // n, db // 2)
    while True:
        p = sieve[i]
        P = p ** k
        if P > half:
            break
        if da % p % 2 and (not db % P):
            return False
        i += 1
    return True
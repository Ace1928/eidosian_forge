from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def primes_appearing(knot_exterior, p):
    """
       sage: M = Manifold('K12n731')
       sage: primes_appearing(M, 3)
       [2, 13]
    """
    C = knot_exterior.covers(p, cover_type='cyclic')[0]
    divisors = C.homology().elementary_divisors()
    primes = set()
    for d in divisors:
        if d != 0:
            primes.update([p for p, e in ZZ(d).factor()])
    return sorted(primes)
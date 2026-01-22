from mpmath.libmp import (fzero, from_int, from_rational,
from sympy.core.numbers import igcd
from .residue_ntheory import (_sqrt_mod_prime_power,
import math
def npartitions(n, verbose=False):
    """
    Calculate the partition function P(n), i.e. the number of ways that
    n can be written as a sum of positive integers.

    P(n) is computed using the Hardy-Ramanujan-Rademacher formula [1]_.


    The correctness of this implementation has been tested through $10^{10}$.

    Examples
    ========

    >>> from sympy.ntheory import npartitions
    >>> npartitions(25)
    1958

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PartitionFunctionP.html

    """
    n = int(n)
    if n < 0:
        return 0
    if n <= 5:
        return [1, 1, 2, 3, 5, 7][n]
    if '_factor' not in globals():
        _pre()
    pbits = int((math.pi * (2 * n / 3.0) ** 0.5 - math.log(4 * n)) / math.log(10) + 1) * math.log(10, 2)
    prec = p = int(pbits * 1.1 + 100)
    s = fzero
    M = max(6, int(0.24 * n ** 0.5 + 4))
    if M > 10 ** 5:
        raise ValueError('Input too big')
    sq23pi = mpf_mul(mpf_sqrt(from_rational(2, 3, p), p), mpf_pi(p), p)
    sqrt8 = mpf_sqrt(from_int(8), p)
    for q in range(1, M):
        a = _a(n, q, p)
        d = _d(n, q, p, sq23pi, sqrt8)
        s = mpf_add(s, mpf_mul(a, d), prec)
        if verbose:
            print('step', q, 'of', M, to_str(a, 10), to_str(d, 10))
        p = bitcount(abs(to_int(d))) + 50
    return int(to_int(mpf_add(s, fhalf, prec)))
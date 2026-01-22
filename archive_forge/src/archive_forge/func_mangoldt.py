from ..libmp.backend import xrange
import math
import cmath
@defun
def mangoldt(ctx, n):
    """
    Evaluates the von Mangoldt function `\\Lambda(n) = \\log p`
    if `n = p^k` a power of a prime, and `\\Lambda(n) = 0` otherwise.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> [mangoldt(n) for n in range(-2,3)]
        [0.0, 0.0, 0.0, 0.0, 0.6931471805599453094172321]
        >>> mangoldt(6)
        0.0
        >>> mangoldt(7)
        1.945910149055313305105353
        >>> mangoldt(8)
        0.6931471805599453094172321
        >>> fsum(mangoldt(n) for n in range(101))
        94.04531122935739224600493
        >>> fsum(mangoldt(n) for n in range(10001))
        10013.39669326311478372032

    """
    n = int(n)
    if n < 2:
        return ctx.zero
    if n % 2 == 0:
        if n & n - 1 == 0:
            return +ctx.ln2
        else:
            return ctx.zero
    for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        if not n % p:
            q, r = (n // p, 0)
            while q > 1:
                q, r = divmod(q, p)
                if r:
                    return ctx.zero
            return ctx.ln(p)
    if ctx.isprime(n):
        return ctx.ln(n)
    if n > 10 ** 30:
        raise NotImplementedError
    k = 2
    while 1:
        p = int(n ** (1.0 / k) + 0.5)
        if p < 2:
            return ctx.zero
        if p ** k == n:
            if ctx.isprime(p):
                return ctx.ln(p)
        k += 1
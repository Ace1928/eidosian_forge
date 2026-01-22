from sympy.core.numbers import igcd
from sympy.core.power import integer_nthroot
from sympy.core.sympify import sympify
from sympy.external.gmpy import HAS_GMPY
from sympy.utilities.misc import as_int
from mpmath.libmp import bitcount as _bitlength
def is_strong_lucas_prp(n):
    """Strong Lucas compositeness test with Selfridge parameters.  Returns
    False if n is definitely composite, and True if n is a strong Lucas
    probable prime.

    This is often used in combination with the Miller-Rabin test, and
    in particular, when combined with M-R base 2 creates the strong BPSW test.

    References
    ==========
    - "Lucas Pseudoprimes", Baillie and Wagstaff, 1980.
      http://mpqs.free.fr/LucasPseudoprimes.pdf
    - OEIS A217255: Strong Lucas Pseudoprimes
      https://oeis.org/A217255
    - https://en.wikipedia.org/wiki/Lucas_pseudoprime
    - https://en.wikipedia.org/wiki/Baillie-PSW_primality_test

    Examples
    ========

    >>> from sympy.ntheory.primetest import isprime, is_strong_lucas_prp
    >>> for i in range(20000):
    ...     if is_strong_lucas_prp(i) and not isprime(i):
    ...        print(i)
    5459
    5777
    10877
    16109
    18971
    """
    from sympy.ntheory.factor_ import trailing
    n = as_int(n)
    if n == 2:
        return True
    if n < 2 or n % 2 == 0:
        return False
    if is_square(n, False):
        return False
    D, P, Q = _lucas_selfridge_params(n)
    if D == 0:
        return False
    s = trailing(n + 1)
    k = n + 1 >> s
    U, V, Qk = _lucas_sequence(n, P, Q, k)
    if U == 0 or V == 0:
        return True
    for r in range(1, s):
        V = (V * V - 2 * Qk) % n
        if V == 0:
            return True
        Qk = pow(Qk, 2, n)
    return False
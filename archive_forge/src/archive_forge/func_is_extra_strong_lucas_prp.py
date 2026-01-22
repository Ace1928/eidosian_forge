from sympy.core.numbers import igcd
from sympy.core.power import integer_nthroot
from sympy.core.sympify import sympify
from sympy.external.gmpy import HAS_GMPY
from sympy.utilities.misc import as_int
from mpmath.libmp import bitcount as _bitlength
def is_extra_strong_lucas_prp(n):
    """Extra Strong Lucas compositeness test.  Returns False if n is
    definitely composite, and True if n is a "extra strong" Lucas probable
    prime.

    The parameters are selected using P = 3, Q = 1, then incrementing P until
    (D|n) == -1.  The test itself is as defined in Grantham 2000, from the
    Mo and Jones preprint.  The parameter selection and test are the same as
    used in OEIS A217719, Perl's Math::Prime::Util, and the Lucas pseudoprime
    page on Wikipedia.

    With these parameters, there are no counterexamples below 2^64 nor any
    known above that range.  It is 20-50% faster than the strong test.

    Because of the different parameters selected, there is no relationship
    between the strong Lucas pseudoprimes and extra strong Lucas pseudoprimes.
    In particular, one is not a subset of the other.

    References
    ==========
    - "Frobenius Pseudoprimes", Jon Grantham, 2000.
      https://www.ams.org/journals/mcom/2001-70-234/S0025-5718-00-01197-2/
    - OEIS A217719: Extra Strong Lucas Pseudoprimes
      https://oeis.org/A217719
    - https://en.wikipedia.org/wiki/Lucas_pseudoprime

    Examples
    ========

    >>> from sympy.ntheory.primetest import isprime, is_extra_strong_lucas_prp
    >>> for i in range(20000):
    ...     if is_extra_strong_lucas_prp(i) and not isprime(i):
    ...        print(i)
    989
    3239
    5777
    10877
    """
    from sympy.ntheory.factor_ import trailing
    n = as_int(n)
    if n == 2:
        return True
    if n < 2 or n % 2 == 0:
        return False
    if is_square(n, False):
        return False
    D, P, Q = _lucas_extrastrong_params(n)
    if D == 0:
        return False
    s = trailing(n + 1)
    k = n + 1 >> s
    U, V, Qk = _lucas_sequence(n, P, Q, k)
    if U == 0 and (V == 2 or V == n - 2):
        return True
    for r in range(1, s):
        if V == 0:
            return True
        V = (V * V - 2) % n
    return False
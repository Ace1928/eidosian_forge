from sympy.core.numbers import igcd
from sympy.core.power import integer_nthroot
from sympy.core.sympify import sympify
from sympy.external.gmpy import HAS_GMPY
from sympy.utilities.misc import as_int
from mpmath.libmp import bitcount as _bitlength
def is_euler_pseudoprime(n, b):
    """Returns True if n is prime or an Euler pseudoprime to base b, else False.

    Euler Pseudoprime : In arithmetic, an odd composite integer n is called an
    euler pseudoprime to base a, if a and n are coprime and satisfy the modular
    arithmetic congruence relation :

    a ^ (n-1)/2 = + 1(mod n) or
    a ^ (n-1)/2 = - 1(mod n)

    (where mod refers to the modulo operation).

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_euler_pseudoprime
    >>> is_euler_pseudoprime(2, 5)
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler_pseudoprime
    """
    from sympy.ntheory.factor_ import trailing
    if not mr(n, [b]):
        return False
    n = as_int(n)
    r = n - 1
    c = pow(b, r >> trailing(r), n)
    if c == 1:
        return True
    while True:
        if c == n - 1:
            return True
        c = pow(c, 2, n)
        if c == 1:
            return False
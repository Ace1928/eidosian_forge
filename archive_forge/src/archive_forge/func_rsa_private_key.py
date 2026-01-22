from string import whitespace, ascii_uppercase as uppercase, printable
from functools import reduce
import warnings
from itertools import cycle
from sympy.core import Symbol
from sympy.core.numbers import igcdex, mod_inverse, igcd, Rational
from sympy.core.random import _randrange, _randint
from sympy.matrices import Matrix
from sympy.ntheory import isprime, primitive_root, factorint
from sympy.ntheory import totient as _euler
from sympy.ntheory import reduced_totient as _carmichael
from sympy.ntheory.generate import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import FF
from sympy.polys.polytools import gcd, Poly
from sympy.utilities.misc import as_int, filldedent, translate
from sympy.utilities.iterables import uniq, multiset
def rsa_private_key(*args, **kwargs):
    """Return the RSA *private key* pair, `(n, d)`

    Parameters
    ==========

    args : naturals
        The keyword is identical to the ``args`` in
        :meth:`rsa_public_key`.

    totient : bool, optional
        If ``'Euler'``, it uses Euler's totient convention `\\phi(n)`
        which is :meth:`sympy.ntheory.factor_.totient` in SymPy.

        If ``'Carmichael'``, it uses Carmichael's totient convention
        `\\lambda(n)` which is
        :meth:`sympy.ntheory.factor_.reduced_totient` in SymPy.

        There can be some output differences for private key generation
        as examples below.

        Example using Euler's totient:

        >>> from sympy.crypto.crypto import rsa_private_key
        >>> rsa_private_key(61, 53, 17, totient='Euler')
        (3233, 2753)

        Example using Carmichael's totient:

        >>> from sympy.crypto.crypto import rsa_private_key
        >>> rsa_private_key(61, 53, 17, totient='Carmichael')
        (3233, 413)

    index : nonnegative integer, optional
        Returns an arbitrary solution of a RSA private key at the index
        specified at `0, 1, 2, \\dots`. This parameter needs to be
        specified along with ``totient='Carmichael'``.

        RSA private exponent is a non-unique solution of
        `e d \\mod \\lambda(n) = 1` and it is possible in any form of
        `d + k \\lambda(n)`, where `d` is an another
        already-computed private exponent, and `\\lambda` is a
        Carmichael's totient function, and `k` is any integer.

        However, considering only the positive cases, there can be
        a principal solution of a RSA private exponent `d_0` in
        `0 < d_0 < \\lambda(n)`, and all the other solutions
        can be canonicalzed in a form of `d_0 + k \\lambda(n)`.

        ``index`` specifies the `k` notation to yield any possible value
        an RSA private key can have.

        An example of computing any arbitrary RSA private key:

        >>> from sympy.crypto.crypto import rsa_private_key
        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=0)
        (3233, 413)
        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=1)
        (3233, 1193)
        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=2)
        (3233, 1973)

    multipower : bool, optional
        The keyword is identical to the ``multipower`` in
        :meth:`rsa_public_key`.

    Returns
    =======

    (n, d) : int, int
        `n` is a product of any arbitrary number of primes given as
        the argument.

        `d` is the inverse of `e` (mod `\\phi(n)`) where `e` is the
        exponent given, and `\\phi` is a Euler totient.

    False
        Returned if less than two arguments are given, or `e` is
        not relatively prime to the totient of the modulus.

    Examples
    ========

    >>> from sympy.crypto.crypto import rsa_private_key

    A private key of a two-prime RSA:

    >>> p, q, e = 3, 5, 7
    >>> rsa_private_key(p, q, e)
    (15, 7)
    >>> rsa_private_key(p, q, 30)
    False

    A private key of a multiprime RSA:

    >>> primes = [2, 3, 5, 7, 11, 13]
    >>> e = 7
    >>> args = primes + [e]
    >>> rsa_private_key(*args)
    (30030, 823)

    See Also
    ========

    rsa_public_key
    encipher_rsa
    decipher_rsa

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/RSA_%28cryptosystem%29

    .. [2] https://cacr.uwaterloo.ca/techreports/2006/cacr2006-16.pdf

    .. [3] https://link.springer.com/content/pdf/10.1007/BFb0055738.pdf

    .. [4] https://www.itiis.org/digital-library/manuscript/1381
    """
    return _rsa_key(*args, public=False, private=True, **kwargs)
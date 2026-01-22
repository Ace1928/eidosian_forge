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
def encipher_rsa(i, key, factors=None):
    """Encrypt the plaintext with RSA.

    Parameters
    ==========

    i : integer
        The plaintext to be encrypted for.

    key : (n, e) where n, e are integers
        `n` is the modulus of the key and `e` is the exponent of the
        key. The encryption is computed by `i^e \\bmod n`.

        The key can either be a public key or a private key, however,
        the message encrypted by a public key can only be decrypted by
        a private key, and vice versa, as RSA is an asymmetric
        cryptography system.

    factors : list of coprime integers
        This is identical to the keyword ``factors`` in
        :meth:`decipher_rsa`.

    Notes
    =====

    Some specifications may make the RSA not cryptographically
    meaningful.

    For example, `0`, `1` will remain always same after taking any
    number of exponentiation, thus, should be avoided.

    Furthermore, if `i^e < n`, `i` may easily be figured out by taking
    `e` th root.

    And also, specifying the exponent as `1` or in more generalized form
    as `1 + k \\lambda(n)` where `k` is an nonnegative integer,
    `\\lambda` is a carmichael totient, the RSA becomes an identity
    mapping.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_rsa
    >>> from sympy.crypto.crypto import rsa_public_key, rsa_private_key

    Public Key Encryption:

    >>> p, q, e = 3, 5, 7
    >>> puk = rsa_public_key(p, q, e)
    >>> msg = 12
    >>> encipher_rsa(msg, puk)
    3

    Private Key Encryption:

    >>> p, q, e = 3, 5, 7
    >>> prk = rsa_private_key(p, q, e)
    >>> msg = 12
    >>> encipher_rsa(msg, prk)
    3

    Encryption using chinese remainder theorem:

    >>> encipher_rsa(msg, prk, factors=[p, q])
    3
    """
    return _encipher_decipher_rsa(i, key, factors=factors)
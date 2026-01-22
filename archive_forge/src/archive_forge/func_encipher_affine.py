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
def encipher_affine(msg, key, symbols=None, _inverse=False):
    """
    Performs the affine cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    Explanation
    ===========

    Encryption is based on the map `x \\rightarrow ax+b` (mod `N`)
    where ``N`` is the number of characters in the alphabet.
    Decryption is based on the map `x \\rightarrow cx+d` (mod `N`),
    where `c = a^{-1}` (mod `N`) and `d = -a^{-1}b` (mod `N`).
    In particular, for the map to be invertible, we need
    `\\mathrm{gcd}(a, N) = 1` and an error will be raised if this is
    not true.

    Parameters
    ==========

    msg : str
        Characters that appear in ``symbols``.

    a, b : int, int
        A pair integers, with ``gcd(a, N) = 1`` (the secret key).

    symbols
        String of characters (default = uppercase letters).

        When no symbols are given, ``msg`` is converted to upper case
        letters and all other characters are ignored.

    Returns
    =======

    ct
        String of characters (the ciphertext message)

    Notes
    =====

    ALGORITHM:

        STEPS:
            0. Number the letters of the alphabet from 0, ..., N
            1. Compute from the string ``msg`` a list ``L1`` of
               corresponding integers.
            2. Compute from the list ``L1`` a new list ``L2``, given by
               replacing ``x`` by ``a*x + b (mod N)``, for each element
               ``x`` in ``L1``.
            3. Compute from the list ``L2`` a string ``ct`` of
               corresponding letters.

    This is a straightforward generalization of the shift cipher with
    the added complexity of requiring 2 characters to be deciphered in
    order to recover the key.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Affine_cipher

    See Also
    ========

    decipher_affine

    """
    msg, _, A = _prep(msg, '', symbols)
    N = len(A)
    a, b = key
    assert gcd(a, N) == 1
    if _inverse:
        c = mod_inverse(a, N)
        d = -b * c
        a, b = (c, d)
    B = ''.join([A[(a * i + b) % N] for i in range(N)])
    return translate(msg, A, B)
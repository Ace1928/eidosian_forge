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
def decipher_bifid5(msg, key):
    """
    Return the Bifid cipher decryption of ``msg``.

    Explanation
    ===========

    This is the version of the Bifid cipher that uses the `5 \\times 5`
    Polybius square; the letter "J" is ignored unless a ``key`` of
    length 25 is used.

    Parameters
    ==========

    msg
        Ciphertext string.

    key
        Short string for key; duplicated characters are ignored and if
        the length is less then 25 characters, it will be padded with
        other letters from the alphabet omitting "J".
        Non-alphabetic characters are ignored.

    Returns
    =======

    plaintext
        Plaintext from Bifid5 cipher (all caps, no spaces).

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_bifid5, decipher_bifid5
    >>> key = "gold bug"
    >>> encipher_bifid5('meet me on friday', key)
    'IEILEHFSTSFXEE'
    >>> encipher_bifid5('meet me on monday', key)
    'IEILHHFSTSFQYE'
    >>> decipher_bifid5(_, key)
    'MEETMEONMONDAY'

    """
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid5)
    key = padded_key(key, bifid5)
    return decipher_bifid(msg, '', key)
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
def encipher_bifid5(msg, key):
    """
    Performs the Bifid cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    Explanation
    ===========

    This is the version of the Bifid cipher that uses the `5 \\times 5`
    Polybius square. The letter "J" is ignored so it must be replaced
    with something else (traditionally an "I") before encryption.

    ALGORITHM: (5x5 case)

        STEPS:
            0. Create the `5 \\times 5` Polybius square ``S`` associated
               to ``key`` as follows:

                a) moving from left-to-right, top-to-bottom,
                   place the letters of the key into a `5 \\times 5`
                   matrix,
                b) if the key has less than 25 letters, add the
                   letters of the alphabet not in the key until the
                   `5 \\times 5` square is filled.

            1. Create a list ``P`` of pairs of numbers which are the
               coordinates in the Polybius square of the letters in
               ``msg``.
            2. Let ``L1`` be the list of all first coordinates of ``P``
               (length of ``L1 = n``), let ``L2`` be the list of all
               second coordinates of ``P`` (so the length of ``L2``
               is also ``n``).
            3. Let ``L`` be the concatenation of ``L1`` and ``L2``
               (length ``L = 2*n``), except that consecutive numbers
               are paired ``(L[2*i], L[2*i + 1])``. You can regard
               ``L`` as a list of pairs of length ``n``.
            4. Let ``C`` be the list of all letters which are of the
               form ``S[i, j]``, for all ``(i, j)`` in ``L``. As a
               string, this is the ciphertext of ``msg``.

    Parameters
    ==========

    msg : str
        Plaintext string.

        Converted to upper case and filtered of anything but all letters
        except J.

    key
        Short string for key; non-alphabetic letters, J and duplicated
        characters are ignored and then, if the length is less than 25
        characters, it is padded with other letters of the alphabet
        (in alphabetical order).

    Returns
    =======

    ct
        Ciphertext (all caps, no spaces).

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     encipher_bifid5, decipher_bifid5)

    "J" will be omitted unless it is replaced with something else:

    >>> round_trip = lambda m, k: \\
    ...     decipher_bifid5(encipher_bifid5(m, k), k)
    >>> key = 'a'
    >>> msg = "JOSIE"
    >>> round_trip(msg, key)
    'OSIE'
    >>> round_trip(msg.replace("J", "I"), key)
    'IOSIE'
    >>> j = "QIQ"
    >>> round_trip(msg.replace("J", j), key).replace(j, "J")
    'JOSIE'


    Notes
    =====

    The Bifid cipher was invented around 1901 by Felix Delastelle.
    It is a *fractional substitution* cipher, where letters are
    replaced by pairs of symbols from a smaller alphabet. The
    cipher uses a `5 \\times 5` square filled with some ordering of the
    alphabet, except that "J" is replaced with "I" (this is a so-called
    Polybius square; there is a `6 \\times 6` analog if you add back in
    "J" and also append onto the usual 26 letter alphabet, the digits
    0, 1, ..., 9).
    According to Helen Gaines' book *Cryptanalysis*, this type of cipher
    was used in the field by the German Army during World War I.

    See Also
    ========

    decipher_bifid5, encipher_bifid

    """
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid5)
    key = padded_key(key, bifid5)
    return encipher_bifid(msg, '', key)
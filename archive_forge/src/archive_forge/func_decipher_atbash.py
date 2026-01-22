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
def decipher_atbash(msg, symbols=None):
    """
    Deciphers a given ``msg`` using Atbash cipher and returns it.

    Explanation
    ===========

    ``decipher_atbash`` is functionally equivalent to ``encipher_atbash``.
    However, it has still been added as a separate function to maintain
    consistency.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_atbash, decipher_atbash
    >>> msg = 'GONAVYBEATARMY'
    >>> encipher_atbash(msg)
    'TLMZEBYVZGZINB'
    >>> decipher_atbash(msg)
    'TLMZEBYVZGZINB'
    >>> encipher_atbash(msg) == decipher_atbash(msg)
    True
    >>> msg == encipher_atbash(encipher_atbash(msg))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Atbash

    See Also
    ========

    encipher_atbash

    """
    return decipher_affine(msg, (25, 25), symbols)
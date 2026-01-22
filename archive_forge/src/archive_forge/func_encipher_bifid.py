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
def encipher_bifid(msg, key, symbols=None):
    """
    Performs the Bifid cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    This is the version of the Bifid cipher that uses an `n \\times n`
    Polybius square.

    Parameters
    ==========

    msg
        Plaintext string.

    key
        Short string for key.

        Duplicate characters are ignored and then it is padded with the
        characters in ``symbols`` that were not in the short key.

    symbols
        `n \\times n` characters defining the alphabet.

        (default is string.printable)

    Returns
    =======

    ciphertext
        Ciphertext using Bifid5 cipher without spaces.

    See Also
    ========

    decipher_bifid, encipher_bifid5, encipher_bifid6

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bifid_cipher

    """
    msg, key, A = _prep(msg, key, symbols, bifid10)
    long_key = ''.join(uniq(key)) or A
    n = len(A) ** 0.5
    if n != int(n):
        raise ValueError('Length of alphabet (%s) is not a square number.' % len(A))
    N = int(n)
    if len(long_key) < N ** 2:
        long_key = list(long_key) + [x for x in A if x not in long_key]
    row_col = {ch: divmod(i, N) for i, ch in enumerate(long_key)}
    r, c = zip(*[row_col[x] for x in msg])
    rc = r + c
    ch = {i: ch for ch, i in row_col.items()}
    rv = ''.join((ch[i] for i in zip(rc[::2], rc[1::2])))
    return rv
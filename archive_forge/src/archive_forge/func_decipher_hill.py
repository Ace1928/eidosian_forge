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
def decipher_hill(msg, key, symbols=None):
    """
    Deciphering is the same as enciphering but using the inverse of the
    key matrix.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_hill, decipher_hill
    >>> from sympy import Matrix

    >>> key = Matrix([[1, 2], [3, 5]])
    >>> encipher_hill("meet me on monday", key)
    'UEQDUEODOCTCWQ'
    >>> decipher_hill(_, key)
    'MEETMEONMONDAY'

    When the length of the plaintext (stripped of invalid characters)
    is not a multiple of the key dimension, extra characters will
    appear at the end of the enciphered and deciphered text. In order to
    decipher the text, those characters must be included in the text to
    be deciphered. In the following, the key has a dimension of 4 but
    the text is 2 short of being a multiple of 4 so two characters will
    be added.

    >>> key = Matrix([[1, 1, 1, 2], [0, 1, 1, 0],
    ...               [2, 2, 3, 4], [1, 1, 0, 1]])
    >>> msg = "ST"
    >>> encipher_hill(msg, key)
    'HJEB'
    >>> decipher_hill(_, key)
    'STQQ'
    >>> encipher_hill(msg, key, pad="Z")
    'ISPK'
    >>> decipher_hill(_, key)
    'STZZ'

    If the last two characters of the ciphertext were ignored in
    either case, the wrong plaintext would be recovered:

    >>> decipher_hill("HD", key)
    'ORMV'
    >>> decipher_hill("IS", key)
    'UIKY'

    See Also
    ========

    encipher_hill

    """
    assert key.is_square
    msg, _, A = _prep(msg, '', symbols)
    map = {c: i for i, c in enumerate(A)}
    C = [map[c] for c in msg]
    N = len(A)
    k = key.cols
    n = len(C)
    m, r = divmod(n, k)
    if r:
        C = C + [0] * (k - r)
        m += 1
    key_inv = key.inv_mod(N)
    rv = ''.join([A[p % N] for j in range(m) for p in list(key_inv * Matrix(k, 1, [C[i] for i in range(k * j, k * (j + 1))]))])
    return rv
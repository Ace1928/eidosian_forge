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
def encipher_hill(msg, key, symbols=None, pad='Q'):
    """
    Return the Hill cipher encryption of ``msg``.

    Explanation
    ===========

    The Hill cipher [1]_, invented by Lester S. Hill in the 1920's [2]_,
    was the first polygraphic cipher in which it was practical
    (though barely) to operate on more than three symbols at once.
    The following discussion assumes an elementary knowledge of
    matrices.

    First, each letter is first encoded as a number starting with 0.
    Suppose your message `msg` consists of `n` capital letters, with no
    spaces. This may be regarded an `n`-tuple M of elements of
    `Z_{26}` (if the letters are those of the English alphabet). A key
    in the Hill cipher is a `k x k` matrix `K`, all of whose entries
    are in `Z_{26}`, such that the matrix `K` is invertible (i.e., the
    linear transformation `K: Z_{N}^k \\rightarrow Z_{N}^k`
    is one-to-one).


    Parameters
    ==========

    msg
        Plaintext message of `n` upper-case letters.

    key
        A `k \\times k` invertible matrix `K`, all of whose entries are
        in `Z_{26}` (or whatever number of symbols are being used).

    pad
        Character (default "Q") to use to make length of text be a
        multiple of ``k``.

    Returns
    =======

    ct
        Ciphertext of upper-case letters.

    Notes
    =====

    ALGORITHM:

        STEPS:
            0. Number the letters of the alphabet from 0, ..., N
            1. Compute from the string ``msg`` a list ``L`` of
               corresponding integers. Let ``n = len(L)``.
            2. Break the list ``L`` up into ``t = ceiling(n/k)``
               sublists ``L_1``, ..., ``L_t`` of size ``k`` (with
               the last list "padded" to ensure its size is
               ``k``).
            3. Compute new list ``C_1``, ..., ``C_t`` given by
               ``C[i] = K*L_i`` (arithmetic is done mod N), for each
               ``i``.
            4. Concatenate these into a list ``C = C_1 + ... + C_t``.
            5. Compute from ``C`` a string ``ct`` of corresponding
               letters. This has length ``k*t``.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hill_cipher
    .. [2] Lester S. Hill, Cryptography in an Algebraic Alphabet,
       The American Mathematical Monthly Vol.36, June-July 1929,
       pp.306-312.

    See Also
    ========

    decipher_hill

    """
    assert key.is_square
    assert len(pad) == 1
    msg, pad, A = _prep(msg, pad, symbols)
    map = {c: i for i, c in enumerate(A)}
    P = [map[c] for c in msg]
    N = len(A)
    k = key.cols
    n = len(P)
    m, r = divmod(n, k)
    if r:
        P = P + [map[pad]] * (k - r)
        m += 1
    rv = ''.join([A[c % N] for j in range(m) for c in list(key * Matrix(k, 1, [P[i] for i in range(k * j, k * (j + 1))]))])
    return rv
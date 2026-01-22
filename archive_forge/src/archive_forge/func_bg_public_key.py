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
def bg_public_key(p, q):
    """
    Calculates public keys from private keys.

    Explanation
    ===========

    The function first checks the validity of
    private keys passed as arguments and
    then returns their product.

    Parameters
    ==========

    p, q
        The private keys.

    Returns
    =======

    N
        The public key.

    """
    p, q = bg_private_key(p, q)
    N = p * q
    return N
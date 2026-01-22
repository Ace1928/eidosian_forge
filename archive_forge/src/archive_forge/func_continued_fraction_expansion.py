import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def continued_fraction_expansion(a, b):
    """
    The continued fraction expansion of a/b.

    >>> continued_fraction_expansion(3141,1000)
    [3, 7, 10, 1, 5, 2]
    """
    if b == 0:
        return []
    if b == 1:
        return [a]
    if b < 0:
        return continued_fraction_expansion(-a, -b)
    q, r = (a // b, a % b)
    if a < 0:
        return [q] + continued_fraction_expansion(r, b)[1:]
    return [q] + continued_fraction_expansion(b, r)
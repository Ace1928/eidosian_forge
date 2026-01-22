import common_z3 as CM_Z3
import ctypes
from .z3 import *
def ehash(v):
    """
    Returns a 'stronger' hash value than the default hash() method.
    The result from hash() is not enough to distinguish between 2
    z3 expressions in some cases.

    Note: the following doctests will fail with Python 2.x as the
    default formatting doesn't match that of 3.x.
    >>> x1 = Bool('x'); x2 = Bool('x'); x3 = Int('x')
    >>> print(x1.hash(), x2.hash(), x3.hash()) #BAD: all same hash values
    783810685 783810685 783810685
    >>> print(ehash(x1), ehash(x2), ehash(x3))
    x_783810685_1 x_783810685_1 x_783810685_2

    """
    if z3_debug():
        assert is_expr(v)
    return '{}_{}_{}'.format(str(v), v.hash(), v.sort_kind())
import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def is_dyad(frac) -> bool:
    """ Test if frac is a nonnegative dyadic fraction or integer.

    Examples
    --------
    >>> is_dyad(Fraction(1,4))
    True
    >>> is_dyad(Fraction(1,3))
    False
    >>> is_dyad(0)
    True
    >>> is_dyad(1)
    True
    >>> is_dyad(-Fraction(1,4))
    False
    >>> is_dyad(Fraction(1,6))
    False

    """
    if isinstance(frac, numbers.Integral) and frac >= 0:
        return True
    elif isinstance(frac, Fraction) and frac >= 0 and is_power2(frac.denominator):
        return True
    else:
        return False
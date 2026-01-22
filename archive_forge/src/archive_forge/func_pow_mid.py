import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def pow_mid(p, max_denom: int=1024):
    """ Return (x,1,t) power tuple

        t <= x^p 1^(1-p)

        user wants the epigraph variable t
    """
    assert 0 < p < 1
    p = Fraction(p).limit_denominator(max_denom)
    return (p, (p, 1 - p))
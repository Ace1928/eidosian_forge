import common_z3 as CM_Z3
import ctypes
from .z3 import *
def is_contradiction(claim, verbose=0):
    """
    >>> x,y=Bools('x y')
    >>> is_contradiction(BoolVal(False))
    True

    >>> is_contradiction(BoolVal(True))
    False

    >>> is_contradiction(x)
    False

    >>> is_contradiction(Implies(x,y))
    False

    >>> is_contradiction(Implies(x,x))
    False

    >>> is_contradiction(And(x,Not(x)))
    True
    """
    return prove(claim=Not(claim), assume=None, verbose=verbose)[0]
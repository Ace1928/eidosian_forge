import numpy as np
from cvxpy.atoms.affine.wraps import skew_symmetric_wrap, symmetric_wrap
from cvxpy.expressions.constants import Constant
def real_canon(expr, real_args, imag_args, real2imag):
    if real_args[0] is None:
        return (0 * imag_args[0], None)
    else:
        return (real_args[0], None)
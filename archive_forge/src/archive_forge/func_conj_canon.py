import numpy as np
from cvxpy.atoms.affine.wraps import skew_symmetric_wrap, symmetric_wrap
from cvxpy.expressions.constants import Constant
def conj_canon(expr, real_args, imag_args, real2imag):
    if imag_args[0] is None:
        imag_arg = None
    else:
        imag_arg = -imag_args[0]
    return (real_args[0], imag_arg)
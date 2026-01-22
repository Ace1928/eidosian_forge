import numpy as np
from cvxpy.atoms.affine.wraps import skew_symmetric_wrap, symmetric_wrap
from cvxpy.expressions.constants import Constant
def separable_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize linear functions that are separable
       in real and imaginary parts.
    """
    if all((val is None for val in imag_args)):
        outputs = (expr.copy(real_args), None)
    elif all((val is None for val in real_args)):
        outputs = (None, expr.copy(imag_args))
    else:
        for idx, real_val in enumerate(real_args):
            if real_val is None:
                real_args[idx] = Constant(np.zeros(imag_args[idx].shape))
            elif imag_args[idx] is None:
                imag_args[idx] = Constant(np.zeros(real_args[idx].shape))
        outputs = (expr.copy(real_args), expr.copy(imag_args))
    return outputs
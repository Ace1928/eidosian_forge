import numpy as np
from cvxpy.constraints.zero import Equality, Zero
from cvxpy.expressions.constants import Constant
def zero_canon(expr, real_args, imag_args, real2imag):
    if imag_args[0] is None:
        return ([expr.copy(real_args)], None)
    imag_cons = [Zero(imag_args[0], constr_id=real2imag[expr.id])]
    if real_args[0] is None:
        return (None, imag_cons)
    else:
        return ([expr.copy(real_args)], imag_cons)
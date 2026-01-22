import copy
import numpy as np
from scipy.signal import fftconvolve
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
def tmul(lin_op, value, is_abs: bool=False):
    """Multiply the transpose of the expression tree by a vector.

    Parameters
    ----------
    lin_op : LinOp
        The root of an expression tree.
    value : NumPy matrix
        The vector to multiply by.
    is_abs : bool, optional
        Multiply by the absolute value of the matrix?

    Returns
    -------
    dict
        A map of variable id to value.
    """
    if lin_op.type is lo.VARIABLE:
        return {lin_op.data: value}
    elif lin_op.type is lo.NO_OP:
        return {}
    else:
        if is_abs:
            result = op_abs_tmul(lin_op, value)
        else:
            result = op_tmul(lin_op, value)
        result_dicts = []
        for arg in lin_op.args:
            result_dicts.append(tmul(arg, result, is_abs))
        return sum_dicts(result_dicts)
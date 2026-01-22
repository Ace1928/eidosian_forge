import copy
import numpy as np
from scipy.signal import fftconvolve
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
def op_abs_mul(lin_op, args):
    """Applies the absolute value of the linear operator to the arguments.

    Parameters
    ----------
    lin_op : LinOp
        A linear operator.
    args : list
        The arguments to the operator.

    Returns
    -------
    NumPy matrix or SciPy sparse matrix.
        The result of applying the linear operator.
    """
    if lin_op.type in [lo.SCALAR_CONST, lo.DENSE_CONST, lo.SPARSE_CONST]:
        result = np.abs(lin_op.data)
    elif lin_op.type is lo.NEG:
        result = args[0]
    elif lin_op.type is lo.MUL:
        coeff = mul(lin_op.data, {}, True)
        result = coeff * args[0]
    elif lin_op.type is lo.DIV:
        divisor = mul(lin_op.data, {}, True)
        result = args[0] / divisor
    elif lin_op.type is lo.CONV:
        result = conv_mul(lin_op, args[0], is_abs=True)
    else:
        result = op_mul(lin_op, args)
    return result
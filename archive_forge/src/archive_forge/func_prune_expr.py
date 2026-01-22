import copy
import numpy as np
from scipy.signal import fftconvolve
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
def prune_expr(lin_op) -> bool:
    """Prunes constant branches from the expression.

    Parameters
    ----------
    lin_op : LinOp
        The root linear operator.

    Returns
    -------
    bool
        Were all the expression's arguments pruned?
    """
    if lin_op.type is lo.VARIABLE:
        return False
    elif lin_op.type in [lo.SCALAR_CONST, lo.DENSE_CONST, lo.SPARSE_CONST, lo.PARAM]:
        return True
    pruned_args = []
    is_constant = True
    for arg in lin_op.args:
        arg_constant = prune_expr(arg)
        if not arg_constant:
            is_constant = False
            pruned_args.append(arg)
    lin_op.args[:] = pruned_args[:]
    return is_constant
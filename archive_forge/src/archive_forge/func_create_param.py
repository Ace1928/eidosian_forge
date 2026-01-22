from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def create_param(shape: Tuple[int, ...], param_id=None):
    """Wraps a parameter.

    Parameters
    ----------
    shape : tuple
        The (rows, cols) dimensions of the operator.

    Returns
    -------
    LinOP
        A LinOp wrapping the parameter.
    """
    if param_id is None:
        param_id = get_id()
    return lo.LinOp(lo.PARAM, shape, [], param_id)
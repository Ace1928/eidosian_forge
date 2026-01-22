from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def upper_tri(operator):
    """Vectorized upper triangular portion of a square matrix.

    Parameters
    ----------
    operator : LinOp
        The matrix operator.

    Returns
    -------
    LinOp
       LinOp representing the vectorized upper triangle.
    """
    entries = operator.shape[0] * operator.shape[1]
    shape = ((entries - operator.shape[0]) // 2, 1)
    return lo.LinOp(lo.UPPER_TRI, shape, [operator], None)
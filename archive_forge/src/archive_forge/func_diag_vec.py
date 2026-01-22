from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def diag_vec(operator, k: int=0):
    """Converts a vector to a diagonal matrix.

    Parameters
    ----------
    operator : LinOp
        The operator to convert to a diagonal matrix.
    k : int
        The offset of the diagonal.

    Returns
    -------
    LinOp
       LinOp representing the diagonal matrix.
    """
    rows = operator.shape[0] + abs(k)
    shape = (rows, rows)
    return lo.LinOp(lo.DIAG_VEC, shape, [operator], k)
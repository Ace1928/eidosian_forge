from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def is_const(operator) -> bool:
    """Returns whether a LinOp is constant.

    Parameters
    ----------
    operator : LinOp
        The LinOp to test.

    Returns
    -------
        True if the LinOp is a constant, False otherwise.
    """
    return operator.type in [lo.SCALAR_CONST, lo.SPARSE_CONST, lo.DENSE_CONST, lo.PARAM]
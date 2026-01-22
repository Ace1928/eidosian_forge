from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def sum_entries(operator, shape: Tuple[int, ...]):
    """Sum the entries of an operator.

    Parameters
    ----------
    expr : LinOp
        The operator to sum the entries of.
    shape : tuple
        The shape of the sum.

    Returns
    -------
    LinOp
        An operator representing the sum.
    """
    return lo.LinOp(lo.SUM_ENTRIES, shape, [operator], None)
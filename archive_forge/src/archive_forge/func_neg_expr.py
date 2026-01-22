from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def neg_expr(operator):
    """Negate an operator.

    Parameters
    ----------
    expr : LinOp
        The operator to be negated.

    Returns
    -------
    LinOp
        The negated operator.
    """
    return lo.LinOp(lo.NEG, operator.shape, [operator], None)
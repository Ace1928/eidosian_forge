from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def get_constr_expr(lh_op, rh_op):
    """Returns the operator in the constraint.
    """
    if rh_op is None:
        return lh_op
    else:
        return sum_expr([lh_op, neg_expr(rh_op)])
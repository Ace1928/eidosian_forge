from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def create_geq(lh_op, rh_op=None, constr_id=None):
    """Creates an internal greater than or equal constraint.

    Parameters
    ----------
    lh_term : LinOp
        The left-hand operator in the >= constraint.
    rh_term : LinOp
        The right-hand operator in the >= constraint.
    constr_id : int
        The id of the CVXPY equality constraint creating the constraint.

    Returns
    -------
    LinLeqConstr
    """
    if rh_op is not None:
        rh_op = neg_expr(rh_op)
    return create_leq(neg_expr(lh_op), rh_op, constr_id)
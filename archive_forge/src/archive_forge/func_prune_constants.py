import copy
import numpy as np
from scipy.signal import fftconvolve
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
def prune_constants(constraints):
    """Returns a new list of constraints with constant terms removed.

    Parameters
    ----------
    constraints : list
        The constraints that form the matrix.

    Returns
    -------
    list
        The pruned constraints.
    """
    pruned_constraints = []
    for constr in constraints:
        constr_type = type(constr)
        expr = copy.deepcopy(constr.expr)
        is_constant = prune_expr(expr)
        if is_constant:
            expr = lo.LinOp(lo.NO_OP, expr.shape, [], None)
        pruned = constr_type(expr, constr.constr_id, constr.shape)
        pruned_constraints.append(pruned)
    return pruned_constraints
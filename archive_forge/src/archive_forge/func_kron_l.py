from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def kron_l(lh_op, rh_op, shape: Tuple[int, ...]):
    """Kronecker product of two matrices, where the left operand is a Variable

    Parameters
    ----------
    lh_op : LinOp
        The left-hand operator in the product.
    rh_op : LinOp
        The right-hand operator in the product.

    Returns
    -------
    LinOp
        A linear operator representing the Kronecker product.
    """
    return lo.LinOp(lo.KRON_L, shape, [lh_op], rh_op)
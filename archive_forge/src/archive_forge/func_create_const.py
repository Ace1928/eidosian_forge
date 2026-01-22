from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def create_const(value, shape: Tuple[int, ...], sparse: bool=False):
    """Wraps a constant.

    Parameters
    ----------
    value : scalar, NumPy matrix, or SciPy sparse matrix.
        The numeric constant to wrap.
    shape : tuple
        The (rows, cols) dimensions of the constant.
    sparse : bool
        Is the constant a SciPy sparse matrix?

    Returns
    -------
    LinOP
        A LinOp wrapping the constant.
    """
    if shape == (1, 1):
        op_type = lo.SCALAR_CONST
        if not np.isscalar(value):
            value = value[0, 0]
    elif sparse:
        op_type = lo.SPARSE_CONST
    else:
        op_type = lo.DENSE_CONST
    return lo.LinOp(op_type, shape, [], value)
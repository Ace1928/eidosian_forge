from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
def get_diff_mat(dim: int, axis: int) -> sp.csc_matrix:
    """Return a sparse matrix representation of first order difference operator.

    Parameters
    ----------
    dim : int
       The length of the matrix dimensions.
    axis : int
       The axis to take the difference along.

    Returns
    -------
    SciPy CSC matrix
        A square matrix representing first order difference.
    """
    val_arr = []
    row_arr = []
    col_arr = []
    for i in range(dim):
        val_arr.append(1.0)
        row_arr.append(i)
        col_arr.append(i)
        if i > 0:
            val_arr.append(-1.0)
            row_arr.append(i)
            col_arr.append(i - 1)
    mat = sp.csc_matrix((val_arr, (row_arr, col_arr)), (dim, dim))
    if axis == 0:
        return mat
    else:
        return mat.T
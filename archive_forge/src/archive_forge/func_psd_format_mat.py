import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
from cvxpy.utilities.versioning import Version
@staticmethod
def psd_format_mat(constr):
    """Return a linear operator to multiply by PSD constraint coefficients.

        Special cases PSD constraints, as SCS expects constraints to be
        imposed on solely the lower triangular part of the variable matrix.
        Moreover, it requires the off-diagonal coefficients to be scaled by
        sqrt(2), and applies to the symmetric part of the constrained expression.
        """
    rows = cols = constr.expr.shape[0]
    entries = rows * (cols + 1) // 2
    row_arr = np.arange(0, entries)
    lower_diag_indices = np.tril_indices(rows)
    col_arr = np.sort(np.ravel_multi_index(lower_diag_indices, (rows, cols), order='F'))
    val_arr = np.zeros((rows, cols))
    val_arr[lower_diag_indices] = np.sqrt(2)
    np.fill_diagonal(val_arr, 1.0)
    val_arr = np.ravel(val_arr, order='F')
    val_arr = val_arr[np.nonzero(val_arr)]
    shape = (entries, rows * cols)
    scaled_lower_tri = sp.csc_matrix((val_arr, (row_arr, col_arr)), shape)
    idx = np.arange(rows * cols)
    val_symm = 0.5 * np.ones(2 * rows * cols)
    K = idx.reshape((rows, cols))
    row_symm = np.append(idx, np.ravel(K, order='F'))
    col_symm = np.append(idx, np.ravel(K.T, order='F'))
    symm_matrix = sp.csc_matrix((val_symm, (row_symm, col_symm)))
    return scaled_lower_tri @ symm_matrix
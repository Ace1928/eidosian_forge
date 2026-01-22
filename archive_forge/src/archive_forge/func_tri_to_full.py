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
def tri_to_full(lower_tri, n):
    """Expands n*(n+1)//2 lower triangular to full matrix

    Scales off-diagonal by 1/sqrt(2), as per the SCS specification.

    Parameters
    ----------
    lower_tri : numpy.ndarray
        A NumPy array representing the lower triangular part of the
        matrix, stacked in column-major order.
    n : int
        The number of rows (columns) in the full square matrix.

    Returns
    -------
    numpy.ndarray
        A 2-dimensional ndarray that is the scaled expansion of the lower
        triangular array.

    Notes
    -----
    SCS tracks "lower triangular" indices in a way that corresponds to numpy's
    "upper triangular" indices. So the function call below uses ``np.triu_indices``
    in a way that looks weird, but is nevertheless correct.
    """
    full = np.zeros((n, n))
    full[np.triu_indices(n)] = lower_tri
    full += full.T
    full[np.diag_indices(n)] /= 2
    full[np.tril_indices(n, k=-1)] /= np.sqrt(2)
    full[np.triu_indices(n, k=1)] /= np.sqrt(2)
    return np.reshape(full, n * n, order='F')
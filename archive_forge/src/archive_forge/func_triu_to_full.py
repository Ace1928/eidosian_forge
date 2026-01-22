import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.versioning import Version
def triu_to_full(upper_tri, n):
    """Expands n*(n+1)//2 upper triangular to full matrix, scaling
    off diagonals by 1/sqrt(2).   This is similar to the SCS behaviour,
    but the upper triangle is used.

    Parameters
    ----------
    upper_tri : numpy.ndarray
        A NumPy array representing the upper triangular part of the
        matrix, stacked in column-major order.
    n : int
        The number of rows (columns) in the full square matrix.

    Returns
    -------
    numpy.ndarray
        A 2-dimensional ndarray that is the scaled expansion of the upper
        triangular array.

    Notes
    -----
    As in the related SCS function, the function below appears to have
    triu/tril confused but is nevertheless correct.

    """
    full = np.zeros((n, n))
    full[np.tril_indices(n)] = upper_tri
    full += full.T
    full[np.diag_indices(n)] /= 2
    full[np.tril_indices(n, k=-1)] /= np.sqrt(2)
    full[np.triu_indices(n, k=1)] /= np.sqrt(2)
    return np.reshape(full, n * n, order='F')
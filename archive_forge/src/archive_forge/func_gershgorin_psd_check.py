import cvxpy.utilities.cpp.sparsecholesky as spchol  # noqa: I001
import cvxpy.settings as settings
import numpy as np
import scipy.linalg as la
import scipy.sparse as spar
import scipy.sparse.linalg as sparla
from scipy.sparse import csc_matrix
def gershgorin_psd_check(A, tol):
    """
    Use the Gershgorin Circle Theorem

        https://en.wikipedia.org/wiki/Gershgorin_circle_theorem

    As a sufficient condition for A being PSD with tolerance "tol".

    The computational complexity of this function is O(nnz(A)).

    Parameters
    ----------
    A : Union[np.ndarray, spar.spmatrix]
        Symmetric (or Hermitian) NumPy ndarray or SciPy sparse matrix.

    tol : float
        Nonnegative. Something very small, like 1e-10.

    Returns
    -------
    True if A is PSD according to the Gershgorin Circle Theorem.
    Otherwise, return False.
    """
    if isinstance(A, spar.spmatrix):
        diag = A.diagonal()
        if np.any(diag < -tol):
            return False
        A_shift = A - spar.diags(diag)
        A_shift = np.abs(A_shift)
        radii = np.array(A_shift.sum(axis=0)).ravel()
        return np.all(diag - radii >= -tol)
    elif isinstance(A, np.ndarray):
        diag = np.diag(A)
        if np.any(diag < -tol):
            return False
        A_shift = A - np.diag(diag)
        A_shift = np.abs(A_shift)
        radii = A_shift.sum(axis=0)
        return np.all(diag - radii >= -tol)
    else:
        raise ValueError()
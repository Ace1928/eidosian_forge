from __future__ import division
import warnings
from typing import Tuple
import numpy as np
import scipy.sparse as sp
from scipy import linalg as LA
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.expression import Expression
from cvxpy.interface.matrix_utilities import is_sparse
from cvxpy.utilities.linalg import sparse_cholesky
def decomp_quad(P, cond=None, rcond=None, lower=True, check_finite: bool=True):
    """
    Compute a matrix decomposition.

    Compute sgn, scale, M such that P = sgn * scale * dot(M, M.T).
    The strategy of determination of eigenvalue negligibility follows
    the pinvh contributions from the scikit-learn project to scipy.

    Parameters
    ----------
    P : matrix or ndarray
        A real symmetric positive or negative (semi)definite input matrix
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue
        are considered negligible.
        If None or -1, suitable machine precision is used (default).
    lower : bool, optional
        Whether the array data is taken from the lower or upper triangle of P.
        The default is to take it from the lower triangle.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        The default is True; disabling may give a performance gain
        but may result in problems (crashes, non-termination) if the inputs
        contain infinities or NaNs.

    Returns
    -------
    scale : float
        induced matrix 2-norm of P
    M1, M2 : 2d ndarray
        A rectangular ndarray such that P = scale * (dot(M1, M1.T) - dot(M2, M2.T))

    """
    if is_sparse(P):
        try:
            sign, L, p = sparse_cholesky(P)
            if sign > 0:
                return (1.0, L[p, :], np.empty((0, 0)))
            else:
                return (1.0, np.empty((0, 0)), L[:, p])
        except ValueError:
            P = np.array(P.todense())
    w, V = LA.eigh(P, lower=lower, check_finite=check_finite)
    if rcond is not None:
        cond = rcond
    if cond in (None, -1):
        t = V.dtype.char.lower()
        factor = {'f': 1000.0, 'd': 1000000.0}
        cond = factor[t] * np.finfo(t).eps
    scale = max(np.absolute(w))
    if scale == 0:
        w_scaled = w
    else:
        w_scaled = w / scale
    maskp = w_scaled > cond
    maskn = w_scaled < -cond
    if np.any(maskp) and np.any(maskn):
        warnings.warn('Forming a nonconvex expression quad_form(x, indefinite).')
    M1 = V[:, maskp] * np.sqrt(w_scaled[maskp])
    M2 = V[:, maskn] * np.sqrt(-w_scaled[maskn])
    return (scale, M1, M2)
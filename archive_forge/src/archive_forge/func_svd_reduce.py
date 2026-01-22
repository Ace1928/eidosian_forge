import sys
import numpy as np
from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
import inspect
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._linesearch import scalar_search_wolfe1, scalar_search_armijo
def svd_reduce(self, max_rank, to_retain=None):
    """
        Reduce the rank of the matrix by retaining some SVD components.

        This corresponds to the "Broyden Rank Reduction Inverse"
        algorithm described in [1]_.

        Note that the SVD decomposition can be done by solving only a
        problem whose size is the effective rank of this matrix, which
        is viable even for large problems.

        Parameters
        ----------
        max_rank : int
            Maximum rank of this matrix after reduction.
        to_retain : int, optional
            Number of SVD components to retain when reduction is done
            (ie. rank > max_rank). Default is ``max_rank - 2``.

        References
        ----------
        .. [1] B.A. van der Rotten, PhD thesis,
           "A limited memory Broyden method to solve high-dimensional
           systems of nonlinear equations". Mathematisch Instituut,
           Universiteit Leiden, The Netherlands (2003).

           https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

        """
    if self.collapsed is not None:
        return
    p = max_rank
    if to_retain is not None:
        q = to_retain
    else:
        q = p - 2
    if self.cs:
        p = min(p, len(self.cs[0]))
    q = max(0, min(q, p - 1))
    m = len(self.cs)
    if m < p:
        return
    C = np.array(self.cs).T
    D = np.array(self.ds).T
    D, R = qr(D, mode='economic')
    C = dot(C, R.T.conj())
    U, S, WH = svd(C, full_matrices=False)
    C = dot(C, inv(WH))
    D = dot(D, WH.T.conj())
    for k in range(q):
        self.cs[k] = C[:, k].copy()
        self.ds[k] = D[:, k].copy()
    del self.cs[q:]
    del self.ds[q:]
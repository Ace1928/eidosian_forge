import operator
from math import prod
import numpy as np
from scipy._lib._util import normalize_axis_index
from scipy.linalg import (get_lapack_funcs, LinAlgError,
from scipy.optimize import minimize_scalar
from . import _bspl
from . import _fitpack_impl
from scipy.sparse import csr_array
from scipy.special import poch
from itertools import combinations
def compute_b_inv(A):
    """
        Inverse 3 central bands of matrix :math:`A=U^T D^{-1} U` assuming that
        ``U`` is a unit upper triangular banded matrix using an algorithm
        proposed in [1].

        Parameters
        ----------
        A : array, shape (4, n)
            Matrix to inverse, stored in LAPACK banded storage.

        Returns
        -------
        B : array, shape (4, n)
            3 unique bands of the symmetric matrix that is an inverse to ``A``.
            The first row is filled with zeros.

        Notes
        -----
        The algorithm is based on the cholesky decomposition and, therefore,
        in case matrix ``A`` is close to not positive defined, the function
        raises LinalgError.

        Both matrices ``A`` and ``B`` are stored in LAPACK banded storage.

        References
        ----------
        .. [1] M. F. Hutchinson and F. R. de Hoog, "Smoothing noisy data with
            spline functions," Numerische Mathematik, vol. 47, no. 1,
            pp. 99-106, 1985.
            :doi:`10.1007/BF01389878`

        """

    def find_b_inv_elem(i, j, U, D, B):
        rng = min(3, n - i - 1)
        rng_sum = 0.0
        if j == 0:
            for k in range(1, rng + 1):
                rng_sum -= U[-k - 1, i + k] * B[-k - 1, i + k]
            rng_sum += D[i]
            B[-1, i] = rng_sum
        else:
            for k in range(1, rng + 1):
                diag = abs(k - j)
                ind = i + min(k, j)
                rng_sum -= U[-k - 1, i + k] * B[-diag - 1, ind + diag]
            B[-j - 1, i + j] = rng_sum
    U = cholesky_banded(A)
    for i in range(2, 5):
        U[-i, i - 1:] /= U[-1, :-i + 1]
    D = 1.0 / U[-1] ** 2
    U[-1] /= U[-1]
    n = U.shape[1]
    B = np.zeros(shape=(4, n))
    for i in range(n - 1, -1, -1):
        for j in range(min(3, n - i - 1), -1, -1):
            find_b_inv_elem(i, j, U, D, B)
    B[0] = [0.0] * n
    return B
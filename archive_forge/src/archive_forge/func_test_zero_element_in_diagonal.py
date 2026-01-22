import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
def test_zero_element_in_diagonal(self):
    """Test if a matrix with a zero diagonal element is singular

        If the i-th diagonal of A is zero, ?tbtrs should return `i` in `info`
        indicating the provided matrix is singular.

        Note that ?tbtrs requires the matrix A to be stored in banded form.
        In this form the diagonal corresponds to the last row."""
    ab = np.ones((3, 4), dtype=float)
    b = np.ones(4, dtype=float)
    tbtrs = get_lapack_funcs('tbtrs', dtype=float)
    ab[-1, 3] = 0
    _, info = tbtrs(ab=ab, b=b, uplo='U')
    assert_equal(info, 4)
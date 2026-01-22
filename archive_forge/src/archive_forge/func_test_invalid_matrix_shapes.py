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
@pytest.mark.parametrize('ldab,n,ldb,nrhs', [(5, 5, 0, 5), (5, 5, 3, 5)])
def test_invalid_matrix_shapes(self, ldab, n, ldb, nrhs):
    """Test ?tbtrs fails correctly if shapes are invalid."""
    ab = np.ones((ldab, n), dtype=float)
    b = np.ones((ldb, nrhs), dtype=float)
    tbtrs = get_lapack_funcs('tbtrs', dtype=float)
    assert_raises(Exception, tbtrs, ab, b)
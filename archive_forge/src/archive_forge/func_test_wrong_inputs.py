import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
def test_wrong_inputs(self):
    assert_raises(ValueError, eigh, np.ones([1, 2]))
    assert_raises(ValueError, eigh, np.ones([2, 2]), np.ones([2, 1]))
    assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([2, 2]))
    assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), type=4)
    assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), subset_by_value=[1, 2], subset_by_index=[2, 4])
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning, "Keyword argument 'eigvals")
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), subset_by_value=[1, 2], eigvals=[2, 4])
    assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), subset_by_index=[0, 4])
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning, "Keyword argument 'eigvals")
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), eigvals=[0, 4])
    assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), subset_by_index=[-2, 2])
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning, "Keyword argument 'eigvals")
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), eigvals=[-2, 2])
    assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), subset_by_index=[2, 0])
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning, "Keyword argument 'eigvals")
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), subset_by_index=[2, 0])
    assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), subset_by_value=[2, 0])
    assert_raises(ValueError, eigh, np.ones([2, 2]), driver='wrong')
    assert_raises(ValueError, eigh, np.ones([3, 3]), None, driver='gvx')
    assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), driver='evr')
    assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), driver='gvd', subset_by_index=[1, 2])
    assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]), driver='gvd', subset_by_index=[1, 2])
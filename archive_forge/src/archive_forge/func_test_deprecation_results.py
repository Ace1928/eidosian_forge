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
def test_deprecation_results(self):
    a = _random_hermitian_matrix(3)
    b = _random_hermitian_matrix(3, posdef=True)
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning, "Keyword argument 'turbo'")
        w_dep, v_dep = eigh(a, b, turbo=True)
    w, v = eigh(a, b, driver='gvd')
    assert_allclose(w_dep, w)
    assert_allclose(v_dep, v)
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning, "Keyword argument 'eigvals'")
        w_dep, v_dep = eigh(a, eigvals=[0, 1])
    w, v = eigh(a, subset_by_index=[0, 1])
    assert_allclose(w_dep, w)
    assert_allclose(v_dep, v)
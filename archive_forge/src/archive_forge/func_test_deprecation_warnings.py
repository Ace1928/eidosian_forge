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
@pytest.mark.parametrize('method', [eigh, eigvalsh])
def test_deprecation_warnings(self, method):
    with pytest.warns(DeprecationWarning, match="Keyword argument 'turbo'"):
        method(np.zeros((2, 2)), turbo=True)
    with pytest.warns(DeprecationWarning, match="Keyword argument 'eigvals'"):
        method(np.zeros((2, 2)), eigvals=[0, 1])
    with pytest.deprecated_call(match='use keyword arguments'):
        method(np.zeros((2, 2)), np.eye(2, 2), True)
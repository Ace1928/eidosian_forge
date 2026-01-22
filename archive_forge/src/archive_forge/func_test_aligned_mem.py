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
@pytest.mark.skipif(platform.machine() == 'ppc64le', reason='crashes on ppc64le')
def test_aligned_mem():
    """Check linalg works with non-aligned memory (float64)"""
    a = arange(804, dtype=np.uint8)
    z = np.frombuffer(a.data, offset=4, count=100, dtype=float)
    z.shape = (10, 10)
    eig(z, overwrite_a=True)
    eig(z.T, overwrite_a=True)
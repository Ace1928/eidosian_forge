import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pytest
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg._onenormest import _onenormest_core, _algorithm_2_2
@pytest.mark.xslow
def test_onenormest_table_3_t_2(self):
    np.random.seed(1234)
    t = 2
    n = 100
    itmax = 5
    nsamples = 5000
    observed = []
    expected = []
    nmult_list = []
    nresample_list = []
    for i in range(nsamples):
        A = scipy.linalg.inv(np.random.randn(n, n))
        est, v, w, nmults, nresamples = _onenormest_core(A, A.T, t, itmax)
        observed.append(est)
        expected.append(scipy.linalg.norm(A, 1))
        nmult_list.append(nmults)
        nresample_list.append(nresamples)
    observed = np.array(observed, dtype=float)
    expected = np.array(expected, dtype=float)
    relative_errors = np.abs(observed - expected) / expected
    underestimation_ratio = observed / expected
    assert_(0.99 < np.mean(underestimation_ratio) < 1.0)
    assert_equal(np.max(nresample_list), 2)
    assert_(0.05 < np.mean(nresample_list) < 0.2)
    nexact = np.count_nonzero(relative_errors < 1e-14)
    proportion_exact = nexact / float(nsamples)
    assert_(0.9 < proportion_exact < 0.95)
    assert_(3.5 < np.mean(nmult_list) < 4.5)
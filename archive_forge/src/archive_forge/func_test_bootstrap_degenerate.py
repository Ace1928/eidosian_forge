import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('method', ['basic', 'percentile', 'BCa'])
def test_bootstrap_degenerate(method):
    data = 35 * [10000.0]
    if method == 'BCa':
        with np.errstate(invalid='ignore'):
            msg = 'The BCa confidence interval cannot be calculated'
            with pytest.warns(stats.DegenerateDataWarning, match=msg):
                res = bootstrap([data], np.mean, method=method)
                assert_equal(res.confidence_interval, (np.nan, np.nan))
    else:
        res = bootstrap([data], np.mean, method=method)
        assert_equal(res.confidence_interval, (10000.0, 10000.0))
    assert_equal(res.standard_error, 0)
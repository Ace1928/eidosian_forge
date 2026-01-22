import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('additional_resamples', [0, 1000])
def test_re_bootstrap(additional_resamples):
    rng = np.random.default_rng(8958153316228384)
    x = rng.random(size=100)
    n1 = 1000
    n2 = additional_resamples
    n3 = n1 + additional_resamples
    rng = np.random.default_rng(296689032789913033)
    res = stats.bootstrap((x,), np.mean, n_resamples=n1, random_state=rng, confidence_level=0.95, method='percentile')
    res = stats.bootstrap((x,), np.mean, n_resamples=n2, random_state=rng, confidence_level=0.9, method='BCa', bootstrap_result=res)
    rng = np.random.default_rng(296689032789913033)
    ref = stats.bootstrap((x,), np.mean, n_resamples=n3, random_state=rng, confidence_level=0.9, method='BCa')
    assert_allclose(res.standard_error, ref.standard_error, rtol=1e-14)
    assert_allclose(res.confidence_interval, ref.confidence_interval, rtol=1e-14)
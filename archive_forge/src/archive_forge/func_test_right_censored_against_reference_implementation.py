import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy import stats
from scipy.stats import _survival
@pytest.mark.parametrize('seed', [182746786639392128, 737379171436494115, 576033618403180168, 308115465002673650])
def test_right_censored_against_reference_implementation(self, seed):
    rng = np.random.default_rng(seed)
    n_unique = rng.integers(10, 100)
    sample, times, censored = self.get_random_sample(rng, n_unique)
    res = stats.ecdf(sample)
    ref = _kaplan_meier_reference(times, censored)
    assert_allclose(res.sf.quantiles, ref[0])
    assert_allclose(res.sf.probabilities, ref[1])
    sample = stats.CensoredData(uncensored=times)
    res = _survival._ecdf_right_censored(sample)
    ref = stats.ecdf(times)
    assert_equal(res[0], ref.sf.quantiles)
    assert_allclose(res[1], ref.cdf.probabilities, rtol=1e-14)
    assert_allclose(res[2], ref.sf.probabilities, rtol=1e-14)
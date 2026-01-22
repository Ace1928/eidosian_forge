import pytest
import numpy as np
import scipy.stats
from ...stats.ecdf_utils import (
@pytest.mark.parametrize('prob', [0.8, 0.9])
@pytest.mark.parametrize('dist', [scipy.stats.norm(3, 10), scipy.stats.poisson(100)], ids=['continuous', 'discrete'])
@pytest.mark.parametrize('ndraws', [10000])
def test_get_pointwise_confidence_band(dist, prob, ndraws, num_trials=1000, seed=57):
    """Test _get_pointwise_confidence_band."""
    eval_points = np.linspace(*dist.interval(0.99), 10)
    cdf_at_eval_points = dist.cdf(eval_points)
    ecdf_lower, ecdf_upper = _get_pointwise_confidence_band(prob, ndraws, cdf_at_eval_points)
    assert np.all(ecdf_lower >= 0)
    assert np.all(ecdf_upper <= 1)
    assert np.all(ecdf_lower <= ecdf_upper)
    in_interval = []
    random_state = np.random.default_rng(seed)
    for _ in range(num_trials):
        ecdf = _simulate_ecdf(ndraws, eval_points, dist.rvs, random_state=random_state)
        in_interval.append((ecdf_lower <= ecdf) & (ecdf < ecdf_upper))
    asymptotic_dist = scipy.stats.norm(np.mean(in_interval, axis=0), scipy.stats.sem(in_interval, axis=0))
    prob_lower, prob_upper = asymptotic_dist.interval(0.999)
    assert np.all(prob_lower <= prob)
    assert np.all(prob <= prob_upper)
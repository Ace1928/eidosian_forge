import math
import numpy as np
from tensorflow.python.ops.distributions import special_math
def test_truncated_normal(assert_equal, assert_all_close, n, y, means=None, stddevs=None, minvals=None, maxvals=None, mean_atol=0.0005, median_atol=0.0008, variance_rtol=0.001):
    """Tests truncated normal distribution's statistics."""

    def _normal_cdf(x):
        return 0.5 * math.erfc(-x / math.sqrt(2))

    def normal_pdf(x):
        return math.exp(-x ** 2 / 2.0) / math.sqrt(2 * math.pi)

    def probit(x):
        return special_math.ndtri(x)
    a = -2.0
    b = 2.0
    mu = 0.0
    sigma = 1.0
    if minvals is not None:
        a = minvals
    if maxvals is not None:
        b = maxvals
    if means is not None:
        mu = means
    if stddevs is not None:
        sigma = stddevs
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    z = _normal_cdf(beta) - _normal_cdf(alpha)
    assert_equal((y >= a).sum(), n)
    assert_equal((y <= b).sum(), n)
    expected_mean = mu + (normal_pdf(alpha) - normal_pdf(beta)) / z * sigma
    y = y.astype(float)
    actual_mean = np.mean(y)
    assert_all_close(actual_mean, expected_mean, atol=mean_atol)
    expected_median = mu + probit((_normal_cdf(alpha) + _normal_cdf(beta)) / 2.0) * sigma
    actual_median = np.median(y)
    assert_all_close(actual_median, expected_median, atol=median_atol)
    expected_variance = sigma ** 2 * (1 + (alpha * normal_pdf(alpha) - beta * normal_pdf(beta)) / z - ((normal_pdf(alpha) - normal_pdf(beta)) / z) ** 2)
    actual_variance = np.var(y)
    assert_all_close(actual_variance, expected_variance, rtol=variance_rtol)
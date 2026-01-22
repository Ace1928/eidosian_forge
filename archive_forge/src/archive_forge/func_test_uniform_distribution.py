import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.murmurhash import murmurhash3_32
def test_uniform_distribution():
    n_bins, n_samples = (10, 100000)
    bins = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_samples):
        bins[murmurhash3_32(i, positive=True) % n_bins] += 1
    means = bins / n_samples
    expected = np.full(n_bins, 1.0 / n_bins)
    assert_array_almost_equal(means / expected, np.ones(n_bins), 2)
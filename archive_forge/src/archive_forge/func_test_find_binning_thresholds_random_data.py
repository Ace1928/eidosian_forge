import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
def test_find_binning_thresholds_random_data():
    bin_thresholds = [_find_binning_thresholds(DATA[:, i], max_bins=255) for i in range(2)]
    for i in range(len(bin_thresholds)):
        assert bin_thresholds[i].shape == (254,)
        assert bin_thresholds[i].dtype == DATA.dtype
    assert_allclose(bin_thresholds[0][[64, 128, 192]], np.array([-0.7, 0.0, 0.7]), atol=0.1)
    assert_allclose(bin_thresholds[1][[64, 128, 192]], np.array([9.99, 10.0, 10.01]), atol=0.01)
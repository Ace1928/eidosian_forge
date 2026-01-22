import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
def test_find_binning_thresholds_low_n_bins():
    bin_thresholds = [_find_binning_thresholds(DATA[:, i], max_bins=128) for i in range(2)]
    for i in range(len(bin_thresholds)):
        assert bin_thresholds[i].shape == (127,)
        assert bin_thresholds[i].dtype == DATA.dtype
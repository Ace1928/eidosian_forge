import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('max_bins', [16, 128, 255])
def test_map_to_bins(max_bins):
    bin_thresholds = [_find_binning_thresholds(DATA[:, i], max_bins=max_bins) for i in range(2)]
    binned = np.zeros_like(DATA, dtype=X_BINNED_DTYPE, order='F')
    is_categorical = np.zeros(2, dtype=np.uint8)
    last_bin_idx = max_bins
    _map_to_bins(DATA, bin_thresholds, is_categorical, last_bin_idx, n_threads, binned)
    assert binned.shape == DATA.shape
    assert binned.dtype == np.uint8
    assert binned.flags.f_contiguous
    min_indices = DATA.argmin(axis=0)
    max_indices = DATA.argmax(axis=0)
    for feature_idx, min_idx in enumerate(min_indices):
        assert binned[min_idx, feature_idx] == 0
    for feature_idx, max_idx in enumerate(max_indices):
        assert binned[max_idx, feature_idx] == max_bins - 1
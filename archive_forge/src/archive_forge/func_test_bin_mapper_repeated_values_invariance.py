import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('n_distinct', [2, 7, 42])
def test_bin_mapper_repeated_values_invariance(n_distinct):
    rng = np.random.RandomState(42)
    distinct_values = rng.normal(size=n_distinct)
    assert len(np.unique(distinct_values)) == n_distinct
    repeated_indices = rng.randint(low=0, high=n_distinct, size=1000)
    data = distinct_values[repeated_indices]
    rng.shuffle(data)
    assert_array_equal(np.unique(data), np.sort(distinct_values))
    data = data.reshape(-1, 1)
    mapper_1 = _BinMapper(n_bins=n_distinct + 1)
    binned_1 = mapper_1.fit_transform(data)
    assert_array_equal(np.unique(binned_1[:, 0]), np.arange(n_distinct))
    mapper_2 = _BinMapper(n_bins=min(256, n_distinct * 3) + 1)
    binned_2 = mapper_2.fit_transform(data)
    assert_allclose(mapper_1.bin_thresholds_[0], mapper_2.bin_thresholds_[0])
    assert_array_equal(binned_1, binned_2)
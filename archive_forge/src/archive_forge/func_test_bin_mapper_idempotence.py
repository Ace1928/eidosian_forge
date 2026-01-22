import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('max_bins_small, max_bins_large', [(2, 2), (3, 3), (4, 4), (42, 42), (255, 255), (5, 17), (42, 255)])
def test_bin_mapper_idempotence(max_bins_small, max_bins_large):
    assert max_bins_large >= max_bins_small
    data = np.random.RandomState(42).normal(size=30000).reshape(-1, 1)
    mapper_small = _BinMapper(n_bins=max_bins_small + 1)
    mapper_large = _BinMapper(n_bins=max_bins_small + 1)
    binned_small = mapper_small.fit_transform(data)
    binned_large = mapper_large.fit_transform(binned_small)
    assert_array_equal(binned_small, binned_large)
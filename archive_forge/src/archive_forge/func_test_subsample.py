import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
def test_subsample():
    mapper_no_subsample = _BinMapper(subsample=None, random_state=0).fit(DATA)
    mapper_subsample = _BinMapper(subsample=256, random_state=0).fit(DATA)
    for feature in range(DATA.shape[1]):
        assert not np.allclose(mapper_no_subsample.bin_thresholds_[feature], mapper_subsample.bin_thresholds_[feature], rtol=0.0001)
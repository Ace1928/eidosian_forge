import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
@pytest.mark.filterwarnings('ignore: Bins whose width are too small')
def test_kbinsdiscretizer_effect_sample_weight():
    """Check the impact of `sample_weight` one computed quantiles."""
    X = np.array([[-2], [-1], [1], [3], [500], [1000]])
    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    est.fit(X, sample_weight=[1, 1, 1, 1, 0, 0])
    assert_allclose(est.bin_edges_[0], [-2, -1, 1, 3])
    assert_allclose(est.transform(X), [[0.0], [1.0], [2.0], [2.0], [2.0], [2.0]])
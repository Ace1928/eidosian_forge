import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
@pytest.mark.parametrize('strategy', ['uniform', 'kmeans', 'quantile'])
def test_same_min_max(strategy):
    warnings.simplefilter('always')
    X = np.array([[1, -2], [1, -1], [1, 0], [1, 1]])
    est = KBinsDiscretizer(strategy=strategy, n_bins=3, encode='ordinal')
    warning_message = 'Feature 0 is constant and will be replaced with 0.'
    with pytest.warns(UserWarning, match=warning_message):
        est.fit(X)
    assert est.n_bins_[0] == 1
    Xt = est.transform(X)
    assert_array_equal(Xt[:, 0], np.zeros(X.shape[0]))
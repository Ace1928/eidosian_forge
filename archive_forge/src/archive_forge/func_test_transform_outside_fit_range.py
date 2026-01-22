import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
@pytest.mark.filterwarnings('ignore:In version 1.5 onwards, subsample=200_000')
@pytest.mark.parametrize('strategy', ['uniform', 'kmeans', 'quantile'])
def test_transform_outside_fit_range(strategy):
    X = np.array([0, 1, 2, 3])[:, None]
    kbd = KBinsDiscretizer(n_bins=4, strategy=strategy, encode='ordinal')
    kbd.fit(X)
    X2 = np.array([-2, 5])[:, None]
    X2t = kbd.transform(X2)
    assert_array_equal(X2t.max(axis=0) + 1, kbd.n_bins_)
    assert_array_equal(X2t.min(axis=0), [0])
import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse, stats
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.feature_selection import (
from sklearn.utils import safe_mask
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_scorefunc_multilabel():
    X = np.array([[10000, 9999, 0], [100, 9999, 0], [1000, 99, 0]])
    y = [[1, 1], [0, 1], [1, 0]]
    Xt = SelectKBest(chi2, k=2).fit_transform(X, y)
    assert Xt.shape == (3, 2)
    assert 0 not in Xt
    Xt = SelectPercentile(chi2, percentile=67).fit_transform(X, y)
    assert Xt.shape == (3, 2)
    assert 0 not in Xt
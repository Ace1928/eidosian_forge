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
def test_tied_pvalues():
    X0 = np.array([[10000, 9999, 9998], [1, 1, 1]])
    y = [0, 1]
    for perm in itertools.permutations((0, 1, 2)):
        X = X0[:, perm]
        Xt = SelectKBest(chi2, k=2).fit_transform(X, y)
        assert Xt.shape == (2, 2)
        assert 9998 not in Xt
        Xt = SelectPercentile(chi2, percentile=67).fit_transform(X, y)
        assert Xt.shape == (2, 2)
        assert 9998 not in Xt
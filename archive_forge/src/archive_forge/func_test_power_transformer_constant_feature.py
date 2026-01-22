import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
@pytest.mark.parametrize('standardize', [True, False])
def test_power_transformer_constant_feature(standardize):
    """Check that PowerTransfomer leaves constant features unchanged."""
    X = [[-2, 0, 2], [-2, 0, 2], [-2, 0, 2]]
    pt = PowerTransformer(method='yeo-johnson', standardize=standardize).fit(X)
    assert_allclose(pt.lambdas_, [1, 1, 1])
    Xft = pt.fit_transform(X)
    Xt = pt.transform(X)
    for Xt_ in [Xft, Xt]:
        if standardize:
            assert_allclose(Xt_, np.zeros_like(X))
        else:
            assert_allclose(Xt_, X)
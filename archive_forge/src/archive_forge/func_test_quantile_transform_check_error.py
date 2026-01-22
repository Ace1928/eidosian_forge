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
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_quantile_transform_check_error(csc_container):
    X = np.transpose([[0, 25, 50, 0, 0, 0, 75, 0, 0, 100], [2, 4, 0, 0, 6, 8, 0, 10, 0, 0], [0, 0, 2.6, 4.1, 0, 0, 2.3, 0, 9.5, 0.1]])
    X = csc_container(X)
    X_neg = np.transpose([[0, 25, 50, 0, 0, 0, 75, 0, 0, 100], [-2, 4, 0, 0, 6, 8, 0, 10, 0, 0], [0, 0, 2.6, 4.1, 0, 0, 2.3, 0, 9.5, 0.1]])
    X_neg = csc_container(X_neg)
    err_msg = 'The number of quantiles cannot be greater than the number of samples used. Got 1000 quantiles and 10 samples.'
    with pytest.raises(ValueError, match=err_msg):
        QuantileTransformer(subsample=10).fit(X)
    transformer = QuantileTransformer(n_quantiles=10)
    err_msg = 'QuantileTransformer only accepts non-negative sparse matrices.'
    with pytest.raises(ValueError, match=err_msg):
        transformer.fit(X_neg)
    transformer.fit(X)
    err_msg = 'QuantileTransformer only accepts non-negative sparse matrices.'
    with pytest.raises(ValueError, match=err_msg):
        transformer.transform(X_neg)
    X_bad_feat = np.transpose([[0, 25, 50, 0, 0, 0, 75, 0, 0, 100], [0, 0, 2.6, 4.1, 0, 0, 2.3, 0, 9.5, 0.1]])
    err_msg = 'X has 2 features, but QuantileTransformer is expecting 3 features as input.'
    with pytest.raises(ValueError, match=err_msg):
        transformer.inverse_transform(X_bad_feat)
    transformer = QuantileTransformer(n_quantiles=10).fit(X)
    with pytest.raises(ValueError, match='Expected 2D array, got scalar array instead'):
        transformer.transform(10)
    transformer = QuantileTransformer(n_quantiles=100)
    warn_msg = 'n_quantiles is set to n_samples'
    with pytest.warns(UserWarning, match=warn_msg) as record:
        transformer.fit(X)
    assert len(record) == 1
    assert transformer.n_quantiles_ == X.shape[0]
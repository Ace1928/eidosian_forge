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
def test_power_transformer_significantly_non_gaussian():
    """Check that significantly non-Gaussian data before transforms correctly.

    For some explored lambdas, the transformed data may be constant and will
    be rejected. Non-regression test for
    https://github.com/scikit-learn/scikit-learn/issues/14959
    """
    X_non_gaussian = 1000000.0 * np.array([0.6, 2.0, 3.0, 4.0] * 4 + [11, 12, 12, 16, 17, 20, 85, 90], dtype=np.float64).reshape(-1, 1)
    pt = PowerTransformer()
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        X_trans = pt.fit_transform(X_non_gaussian)
    assert not np.any(np.isnan(X_trans))
    assert X_trans.mean() == pytest.approx(0.0)
    assert X_trans.std() == pytest.approx(1.0)
    assert X_trans.min() > -2
    assert X_trans.max() < 2
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
def test_robust_scaler_unit_variance():
    rng = np.random.RandomState(42)
    X = rng.randn(1000000, 1)
    X_with_outliers = np.vstack([X, np.ones((100, 1)) * 100, np.ones((100, 1)) * -100])
    quantile_range = (1, 99)
    robust_scaler = RobustScaler(quantile_range=quantile_range, unit_variance=True).fit(X_with_outliers)
    X_trans = robust_scaler.transform(X)
    assert robust_scaler.center_ == pytest.approx(0, abs=0.001)
    assert robust_scaler.scale_ == pytest.approx(1, abs=0.01)
    assert X_trans.std() == pytest.approx(1, abs=0.01)
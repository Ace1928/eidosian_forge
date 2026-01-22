import io
import re
import warnings
from itertools import product
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import kstest
from sklearn import tree
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.impute._base import _most_frequent
from sklearn.linear_model import ARDRegression, BayesianRidge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_union
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_simple_imputer_constant_fill_value_casting():
    """Check that we raise a proper error message when we cannot cast the fill value
    to the input data type. Otherwise, check that the casting is done properly.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28309
    """
    fill_value = 1.5
    X_int64 = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64)
    imputer = SimpleImputer(strategy='constant', fill_value=fill_value, missing_values=2)
    err_msg = f'fill_value={fill_value!r} (of type {type(fill_value)!r}) cannot be cast'
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        imputer.fit(X_int64)
    X_float64 = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float64)
    imputer.fit(X_float64)
    err_msg = f'The dtype of the filling value (i.e. {imputer.statistics_.dtype!r}) cannot be cast'
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        imputer.transform(X_int64)
    fill_value_list = [np.float64(1.5), 1.5, 1]
    X_float32 = X_float64.astype(np.float32)
    for fill_value in fill_value_list:
        imputer = SimpleImputer(strategy='constant', fill_value=fill_value, missing_values=2)
        X_trans = imputer.fit_transform(X_float32)
        assert X_trans.dtype == X_float32.dtype
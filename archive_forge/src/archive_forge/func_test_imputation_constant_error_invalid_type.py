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
@pytest.mark.parametrize('X_data, missing_value', [(1, 0), (1.0, np.nan)])
def test_imputation_constant_error_invalid_type(X_data, missing_value):
    X = np.full((3, 5), X_data, dtype=float)
    X[0, 0] = missing_value
    fill_value = 'x'
    err_msg = f'fill_value={fill_value!r} (of type {type(fill_value)!r}) cannot be cast'
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        imputer = SimpleImputer(missing_values=missing_value, strategy='constant', fill_value=fill_value)
        imputer.fit_transform(X)
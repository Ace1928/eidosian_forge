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
@pytest.mark.parametrize('array_type', ['array', 'sparse'])
@pytest.mark.parametrize('keep_empty_features', [True, False])
def test_simple_imputer_constant_keep_empty_features(array_type, keep_empty_features):
    """Check the behaviour of `keep_empty_features` with `strategy='constant'.
    For backward compatibility, a column full of missing values will always be
    fill and never dropped.
    """
    X = np.array([[np.nan, 2], [np.nan, 3], [np.nan, 6]])
    X = _convert_container(X, array_type)
    fill_value = 10
    imputer = SimpleImputer(strategy='constant', fill_value=fill_value, keep_empty_features=keep_empty_features)
    for method in ['fit_transform', 'transform']:
        X_imputed = getattr(imputer, method)(X)
        assert X_imputed.shape == X.shape
        constant_feature = X_imputed[:, 0].toarray() if array_type == 'sparse' else X_imputed[:, 0]
        assert_array_equal(constant_feature, fill_value)
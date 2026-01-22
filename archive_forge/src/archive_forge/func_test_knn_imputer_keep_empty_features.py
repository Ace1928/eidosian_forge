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
@pytest.mark.parametrize('keep_empty_features', [True, False])
def test_knn_imputer_keep_empty_features(keep_empty_features):
    """Check the behaviour of `keep_empty_features` for `KNNImputer`."""
    X = np.array([[1, np.nan, 2], [3, np.nan, np.nan]])
    imputer = KNNImputer(keep_empty_features=keep_empty_features)
    for method in ['fit_transform', 'transform']:
        X_imputed = getattr(imputer, method)(X)
        if keep_empty_features:
            assert X_imputed.shape == X.shape
            assert_array_equal(X_imputed[:, 1], 0)
        else:
            assert X_imputed.shape == (X.shape[0], X.shape[1] - 1)
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
@pytest.mark.parametrize('min_value, max_value, correct_output', [(0, 100, np.array([[0] * 3, [100] * 3])), (None, None, np.array([[-np.inf] * 3, [np.inf] * 3])), (-np.inf, np.inf, np.array([[-np.inf] * 3, [np.inf] * 3])), ([-5, 5, 10], [100, 200, 300], np.array([[-5, 5, 10], [100, 200, 300]])), ([-5, -np.inf, 10], [100, 200, np.inf], np.array([[-5, -np.inf, 10], [100, 200, np.inf]]))], ids=['scalars', 'None-default', 'inf', 'lists', 'lists-with-inf'])
def test_iterative_imputer_min_max_array_like(min_value, max_value, correct_output):
    X = np.random.RandomState(0).randn(10, 3)
    imputer = IterativeImputer(min_value=min_value, max_value=max_value)
    imputer.fit(X)
    assert isinstance(imputer._min_value, np.ndarray) and isinstance(imputer._max_value, np.ndarray)
    assert imputer._min_value.shape[0] == X.shape[1] and imputer._max_value.shape[0] == X.shape[1]
    assert_allclose(correct_output[0, :], imputer._min_value)
    assert_allclose(correct_output[1, :], imputer._max_value)
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
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_imputation_most_frequent(csc_container):
    X = np.array([[-1, -1, 0, 5], [-1, 2, -1, 3], [-1, 1, 3, -1], [-1, 2, 3, 7]])
    X_true = np.array([[2, 0, 5], [2, 3, 3], [1, 3, 3], [2, 3, 7]])
    _check_statistics(X, X_true, 'most_frequent', [np.nan, 2, 3, 3], -1, csc_container)
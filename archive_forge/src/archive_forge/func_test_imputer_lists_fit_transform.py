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
def test_imputer_lists_fit_transform():
    """Check transform uses object dtype when fitted on an object dtype.

    Non-regression test for #19572.
    """
    X = [['a', 'b'], ['c', 'b'], ['a', 'a']]
    imp_frequent = SimpleImputer(strategy='most_frequent').fit(X)
    X_trans = imp_frequent.transform([[np.nan, np.nan]])
    assert X_trans.dtype == object
    assert_array_equal(X_trans, [['a', 'b']])
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
def test_imputation_copy():
    X_orig = _sparse_random_matrix(5, 5, density=0.75, random_state=0)
    X = X_orig.copy().toarray()
    imputer = SimpleImputer(missing_values=0, strategy='mean', copy=True)
    Xt = imputer.fit(X).transform(X)
    Xt[0, 0] = -1
    assert not np.all(X == Xt)
    X = X_orig.copy()
    imputer = SimpleImputer(missing_values=X.data[0], strategy='mean', copy=True)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert not np.all(X.data == Xt.data)
    X = X_orig.copy().toarray()
    imputer = SimpleImputer(missing_values=0, strategy='mean', copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt[0, 0] = -1
    assert_array_almost_equal(X, Xt)
    X = X_orig.copy().tocsc()
    imputer = SimpleImputer(missing_values=X.data[0], strategy='mean', copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert_array_almost_equal(X.data, Xt.data)
    X = X_orig.copy()
    imputer = SimpleImputer(missing_values=X.data[0], strategy='mean', copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert not np.all(X.data == Xt.data)
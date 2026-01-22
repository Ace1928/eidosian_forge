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
@pytest.mark.parametrize('marker', [None, np.nan, 'NAN', '', 0])
def test_imputation_most_frequent_objects(marker):
    X = np.array([[marker, marker, 'a', 'f'], [marker, 'c', marker, 'd'], [marker, 'b', 'd', marker], [marker, 'c', 'd', 'h']], dtype=object)
    X_true = np.array([['c', 'a', 'f'], ['c', 'd', 'd'], ['b', 'd', 'd'], ['c', 'd', 'h']], dtype=object)
    imputer = SimpleImputer(missing_values=marker, strategy='most_frequent')
    X_trans = imputer.fit(X).transform(X)
    assert_array_equal(X_trans, X_true)
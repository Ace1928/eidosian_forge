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
def test_imputation_constant_object(marker):
    X = np.array([[marker, 'a', 'b', marker], ['c', marker, 'd', marker], ['e', 'f', marker, marker], ['g', 'h', 'i', marker]], dtype=object)
    X_true = np.array([['missing', 'a', 'b', 'missing'], ['c', 'missing', 'd', 'missing'], ['e', 'f', 'missing', 'missing'], ['g', 'h', 'i', 'missing']], dtype=object)
    imputer = SimpleImputer(missing_values=marker, strategy='constant', fill_value='missing')
    X_trans = imputer.fit_transform(X)
    assert_array_equal(X_trans, X_true)
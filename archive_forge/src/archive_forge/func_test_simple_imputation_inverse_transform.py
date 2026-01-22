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
@pytest.mark.parametrize('missing_value', [-1, np.nan])
def test_simple_imputation_inverse_transform(missing_value):
    X_1 = np.array([[9, missing_value, 3, -1], [4, -1, 5, 4], [6, 7, missing_value, -1], [8, 9, 0, missing_value]])
    X_2 = np.array([[5, 4, 2, 1], [2, 1, missing_value, 3], [9, missing_value, 7, 1], [6, 4, 2, missing_value]])
    X_3 = np.array([[1, missing_value, 5, 9], [missing_value, 4, missing_value, missing_value], [2, missing_value, 7, missing_value], [missing_value, 3, missing_value, 8]])
    X_4 = np.array([[1, 1, 1, 3], [missing_value, 2, missing_value, 1], [2, 3, 3, 4], [missing_value, 4, missing_value, 2]])
    imputer = SimpleImputer(missing_values=missing_value, strategy='mean', add_indicator=True)
    X_1_trans = imputer.fit_transform(X_1)
    X_1_inv_trans = imputer.inverse_transform(X_1_trans)
    X_2_trans = imputer.transform(X_2)
    X_2_inv_trans = imputer.inverse_transform(X_2_trans)
    assert_array_equal(X_1_inv_trans, X_1)
    assert_array_equal(X_2_inv_trans, X_2)
    for X in [X_3, X_4]:
        X_trans = imputer.fit_transform(X)
        X_inv_trans = imputer.inverse_transform(X_trans)
        assert_array_equal(X_inv_trans, X)
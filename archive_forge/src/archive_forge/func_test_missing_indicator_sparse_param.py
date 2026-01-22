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
@pytest.mark.parametrize('param_sparse', [True, False, 'auto'])
@pytest.mark.parametrize('arr_type, missing_values', [(np.array, 0)] + list(product(CSC_CONTAINERS + CSR_CONTAINERS + COO_CONTAINERS + LIL_CONTAINERS + BSR_CONTAINERS, [np.nan])))
def test_missing_indicator_sparse_param(arr_type, missing_values, param_sparse):
    X_fit = np.array([[missing_values, missing_values, 1], [4, missing_values, 2]])
    X_trans = np.array([[missing_values, missing_values, 1], [4, 12, 10]])
    X_fit = arr_type(X_fit).astype(np.float64)
    X_trans = arr_type(X_trans).astype(np.float64)
    indicator = MissingIndicator(missing_values=missing_values, sparse=param_sparse)
    X_fit_mask = indicator.fit_transform(X_fit)
    X_trans_mask = indicator.transform(X_trans)
    if param_sparse is True:
        assert X_fit_mask.format == 'csc'
        assert X_trans_mask.format == 'csc'
    elif param_sparse == 'auto' and missing_values == 0:
        assert isinstance(X_fit_mask, np.ndarray)
        assert isinstance(X_trans_mask, np.ndarray)
    elif param_sparse is False:
        assert isinstance(X_fit_mask, np.ndarray)
        assert isinstance(X_trans_mask, np.ndarray)
    elif sparse.issparse(X_fit):
        assert X_fit_mask.format == 'csc'
        assert X_trans_mask.format == 'csc'
    else:
        assert isinstance(X_fit_mask, np.ndarray)
        assert isinstance(X_trans_mask, np.ndarray)
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
@pytest.mark.parametrize('strategy', ['constant', 'most_frequent'])
@pytest.mark.parametrize('dtype', [str, np.dtype('U'), np.dtype('S')])
def test_imputation_const_mostf_error_invalid_types(strategy, dtype):
    X = np.array([[np.nan, np.nan, 'a', 'f'], [np.nan, 'c', np.nan, 'd'], [np.nan, 'b', 'd', np.nan], [np.nan, 'c', 'd', 'h']], dtype=dtype)
    err_msg = 'SimpleImputer does not support data'
    with pytest.raises(ValueError, match=err_msg):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit(X).transform(X)
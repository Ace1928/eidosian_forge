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
def test_simple_impute_pd_na():
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'feature': pd.Series(['abc', None, 'de'], dtype='string')})
    imputer = SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value='na')
    _assert_array_equal_and_same_dtype(imputer.fit_transform(df), np.array([['abc'], ['na'], ['de']], dtype=object))
    df = pd.DataFrame({'feature': pd.Series(['abc', 'de', 'fgh'], dtype='string')})
    imputer = SimpleImputer(fill_value='ok', strategy='constant')
    _assert_array_equal_and_same_dtype(imputer.fit_transform(df), np.array([['abc'], ['de'], ['fgh']], dtype=object))
    df = pd.DataFrame({'feature': pd.Series([1, None, 3], dtype='Int64')})
    imputer = SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value=-1)
    _assert_allclose_and_same_dtype(imputer.fit_transform(df), np.array([[1], [-1], [3]], dtype='float64'))
    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
    _assert_allclose_and_same_dtype(imputer.fit_transform(df), np.array([[1], [-1], [3]], dtype='float64'))
    df = pd.DataFrame({'feature': pd.Series([1, None, 2, 3], dtype='Int64')})
    imputer = SimpleImputer(missing_values=pd.NA, strategy='median')
    _assert_allclose_and_same_dtype(imputer.fit_transform(df), np.array([[1], [2], [2], [3]], dtype='float64'))
    df = pd.DataFrame({'feature': pd.Series([1, None, 2], dtype='Int64')})
    imputer = SimpleImputer(missing_values=pd.NA, strategy='mean')
    _assert_allclose_and_same_dtype(imputer.fit_transform(df), np.array([[1], [1.5], [2]], dtype='float64'))
    df = pd.DataFrame({'feature': pd.Series([1.0, None, 3.0], dtype='float64')})
    imputer = SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value=-2.0)
    _assert_allclose_and_same_dtype(imputer.fit_transform(df), np.array([[1.0], [-2.0], [3.0]], dtype='float64'))
    df = pd.DataFrame({'feature': pd.Series([1.0, None, 2.0, 3.0], dtype='float64')})
    imputer = SimpleImputer(missing_values=pd.NA, strategy='median')
    _assert_allclose_and_same_dtype(imputer.fit_transform(df), np.array([[1.0], [2.0], [2.0], [3.0]], dtype='float64'))
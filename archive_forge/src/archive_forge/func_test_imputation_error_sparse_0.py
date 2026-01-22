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
@pytest.mark.parametrize('strategy', ['mean', 'median', 'most_frequent', 'constant'])
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_imputation_error_sparse_0(strategy, csc_container):
    X = np.ones((3, 5))
    X[0] = 0
    X = csc_container(X)
    imputer = SimpleImputer(strategy=strategy, missing_values=0)
    with pytest.raises(ValueError, match='Provide a dense array'):
        imputer.fit(X)
    imputer.fit(X.toarray())
    with pytest.raises(ValueError, match='Provide a dense array'):
        imputer.transform(X)
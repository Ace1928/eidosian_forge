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
@pytest.mark.parametrize('order, idx_order', [('ascending', [3, 4, 2, 0, 1]), ('descending', [1, 0, 2, 4, 3])])
def test_imputation_order(order, idx_order):
    rng = np.random.RandomState(42)
    X = rng.rand(100, 5)
    X[:50, 1] = np.nan
    X[:30, 0] = np.nan
    X[:20, 2] = np.nan
    X[:10, 4] = np.nan
    with pytest.warns(ConvergenceWarning):
        trs = IterativeImputer(max_iter=1, imputation_order=order, random_state=0).fit(X)
        idx = [x.feat_idx for x in trs.imputation_sequence_]
        assert idx == idx_order
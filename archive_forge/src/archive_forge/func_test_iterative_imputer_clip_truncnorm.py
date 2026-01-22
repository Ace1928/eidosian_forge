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
def test_iterative_imputer_clip_truncnorm():
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    X = _sparse_random_matrix(n, d, density=0.1, random_state=rng).toarray()
    X[:, 0] = 1
    imputer = IterativeImputer(missing_values=0, max_iter=2, n_nearest_features=5, sample_posterior=True, min_value=0.1, max_value=0.2, verbose=1, imputation_order='random', random_state=rng)
    Xt = imputer.fit_transform(X)
    assert_allclose(np.min(Xt[X == 0]), 0.1)
    assert_allclose(np.max(Xt[X == 0]), 0.2)
    assert_allclose(Xt[X != 0], X[X != 0])
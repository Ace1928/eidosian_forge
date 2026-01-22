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
def test_iterative_imputer_truncated_normal_posterior():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(5, 5))
    X[0][0] = np.nan
    imputer = IterativeImputer(min_value=0, max_value=0.5, sample_posterior=True, random_state=rng)
    imputer.fit_transform(X)
    imputations = np.array([imputer.transform(X)[0][0] for _ in range(100)])
    assert all(imputations >= 0)
    assert all(imputations <= 0.5)
    mu, sigma = (imputations.mean(), imputations.std())
    ks_statistic, p_value = kstest((imputations - mu) / sigma, 'norm')
    if sigma == 0:
        sigma += 1e-12
    ks_statistic, p_value = kstest((imputations - mu) / sigma, 'norm')
    assert ks_statistic < 0.2 or p_value > 0.1, 'The posterior does appear to be normal'
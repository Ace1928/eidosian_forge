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
@pytest.mark.parametrize('estimator', [None, DummyRegressor(), BayesianRidge(), ARDRegression(), RidgeCV()])
def test_iterative_imputer_estimators(estimator):
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    X = _sparse_random_matrix(n, d, density=0.1, random_state=rng).toarray()
    imputer = IterativeImputer(missing_values=0, max_iter=1, estimator=estimator, random_state=rng)
    imputer.fit_transform(X)
    hashes = []
    for triplet in imputer.imputation_sequence_:
        expected_type = type(estimator) if estimator is not None else type(BayesianRidge())
        assert isinstance(triplet.estimator, expected_type)
        hashes.append(id(triplet.estimator))
    assert len(set(hashes)) == len(hashes)
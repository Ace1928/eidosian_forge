import warnings
from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._ridge import (
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
from sklearn.model_selection import (
from sklearn.preprocessing import minmax_scale
from sklearn.utils import _IS_32BIT, check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('scoring', [None, 'neg_mean_squared_error', _mean_squared_error_callable])
def test_ridgecv_store_cv_values(scoring):
    rng = np.random.RandomState(42)
    n_samples = 8
    n_features = 5
    x = rng.randn(n_samples, n_features)
    alphas = [0.1, 1.0, 10.0]
    n_alphas = len(alphas)
    scoring_ = make_scorer(scoring) if callable(scoring) else scoring
    r = RidgeCV(alphas=alphas, cv=None, store_cv_values=True, scoring=scoring_)
    y = rng.randn(n_samples)
    r.fit(x, y)
    assert r.cv_values_.shape == (n_samples, n_alphas)
    n_targets = 3
    y = rng.randn(n_samples, n_targets)
    r.fit(x, y)
    assert r.cv_values_.shape == (n_samples, n_targets, n_alphas)
    r = RidgeCV(cv=3, store_cv_values=True, scoring=scoring)
    with pytest.raises(ValueError, match='cv!=None and store_cv_values'):
        r.fit(x, y)
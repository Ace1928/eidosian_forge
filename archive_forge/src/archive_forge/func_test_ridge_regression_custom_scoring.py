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
@pytest.mark.parametrize('cv', [None, KFold(5)])
@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS)
def test_ridge_regression_custom_scoring(sparse_container, cv):

    def _dummy_score(y_test, y_pred):
        return 0.42
    X = X_iris if sparse_container is None else sparse_container(X_iris)
    alphas = np.logspace(-2, 2, num=5)
    clf = RidgeClassifierCV(alphas=alphas, scoring=make_scorer(_dummy_score), cv=cv)
    clf.fit(X, y_iris)
    assert clf.best_score_ == pytest.approx(0.42)
    assert clf.alpha_ == pytest.approx(alphas[0])
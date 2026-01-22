import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', [None, *CSR_CONTAINERS])
@pytest.mark.parametrize('loss', ('epsilon_insensitive', 'squared_epsilon_insensitive'))
def test_regressor_correctness(loss, csr_container):
    y_bin = y.copy()
    y_bin[y != 1] = -1
    reg1 = MyPassiveAggressive(loss=loss, n_iter=2)
    reg1.fit(X, y_bin)
    data = csr_container(X) if csr_container is not None else X
    reg2 = PassiveAggressiveRegressor(tol=None, loss=loss, max_iter=2, shuffle=False)
    reg2.fit(data, y_bin)
    assert_array_almost_equal(reg1.w, reg2.coef_.ravel(), decimal=2)
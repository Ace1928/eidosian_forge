import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('average', [False, True])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('csr_container', [None, *CSR_CONTAINERS])
def test_regressor_mse(csr_container, fit_intercept, average):
    y_bin = y.copy()
    y_bin[y != 1] = -1
    data = csr_container(X) if csr_container is not None else X
    reg = PassiveAggressiveRegressor(C=1.0, fit_intercept=fit_intercept, random_state=0, average=average, max_iter=5)
    reg.fit(data, y_bin)
    pred = reg.predict(data)
    assert np.mean((pred - y_bin) ** 2) < 1.7
    if average:
        assert hasattr(reg, '_average_coef')
        assert hasattr(reg, '_average_intercept')
        assert hasattr(reg, '_standard_intercept')
        assert hasattr(reg, '_standard_coef')
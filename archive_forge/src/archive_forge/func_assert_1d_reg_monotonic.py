import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.tree import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS
def assert_1d_reg_monotonic(clf, monotonic_sign, min_x, max_x, n_steps):
    X_grid = np.linspace(min_x, max_x, n_steps).reshape(-1, 1)
    y_pred_grid = clf.predict(X_grid)
    if monotonic_sign == 1:
        assert (np.diff(y_pred_grid) >= 0.0).all()
    elif monotonic_sign == -1:
        assert (np.diff(y_pred_grid) <= 0.0).all()
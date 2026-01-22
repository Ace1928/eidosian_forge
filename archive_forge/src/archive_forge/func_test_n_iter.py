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
@ignore_warnings
def test_n_iter():
    n_targets = 2
    X, y = (X_diabetes, y_diabetes)
    y_n = np.tile(y, (n_targets, 1)).T
    for max_iter in range(1, 4):
        for solver in ('sag', 'saga', 'lsqr'):
            reg = Ridge(solver=solver, max_iter=max_iter, tol=1e-12)
            reg.fit(X, y_n)
            assert_array_equal(reg.n_iter_, np.tile(max_iter, n_targets))
    for solver in ('sparse_cg', 'svd', 'cholesky'):
        reg = Ridge(solver=solver, max_iter=1, tol=0.1)
        reg.fit(X, y_n)
        assert reg.n_iter_ is None
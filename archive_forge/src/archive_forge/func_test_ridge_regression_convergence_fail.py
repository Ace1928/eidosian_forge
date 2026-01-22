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
def test_ridge_regression_convergence_fail():
    rng = np.random.RandomState(0)
    y = rng.randn(5)
    X = rng.randn(5, 10)
    warning_message = 'sparse_cg did not converge after [0-9]+ iterations.'
    with pytest.warns(ConvergenceWarning, match=warning_message):
        ridge_regression(X, y, alpha=1.0, solver='sparse_cg', tol=0.0, max_iter=None, verbose=1)
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
@pytest.mark.parametrize('n_col', [(), (1,), (3,)])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_X_CenterStackOp(n_col, csr_container):
    rng = np.random.RandomState(0)
    X = rng.randn(11, 8)
    X_m = rng.randn(8)
    sqrt_sw = rng.randn(len(X))
    Y = rng.randn(11, *n_col)
    A = rng.randn(9, *n_col)
    operator = _X_CenterStackOp(csr_container(X), X_m, sqrt_sw)
    reference_operator = np.hstack([X - sqrt_sw[:, None] * X_m, sqrt_sw[:, None]])
    assert_allclose(reference_operator.dot(A), operator.dot(A))
    assert_allclose(reference_operator.T.dot(Y), operator.T.dot(Y))
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
@pytest.mark.parametrize('test_func', (_test_ridge_loo, _test_ridge_cv, _test_ridge_diabetes, _test_multi_ridge_diabetes, _test_ridge_classifiers, _test_tolerance))
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_dense_sparse(test_func, csr_container):
    ret_dense = test_func(None)
    ret_sparse = test_func(csr_container)
    if ret_dense is not None and ret_sparse is not None:
        assert_array_almost_equal(ret_dense, ret_sparse, decimal=3)
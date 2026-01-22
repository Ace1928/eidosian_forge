import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
def test_reorder_vector():
    nobs = 5
    k_endog = 3
    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1
    given = np.zeros((k_endog, nobs))
    given[:, :] = np.array([1, 2, 3])[:, np.newaxis]
    desired = given.copy()
    given[:, 0] = [2, 3, 0]
    desired[:, 0] = [0, 2, 3]
    given[:, 1] = [3, 0, 0]
    desired[:, 1] = [0, 0, 3]
    given[:, 2] = [2, 0, 0]
    desired[:, 2] = [0, 2, 0]
    given[:, 3] = [1, 3, 0]
    desired[:, 3] = [1, 0, 3]
    given[:, 4] = [1, 2, 0]
    desired[:, 4] = [1, 2, 0]
    actual = np.asfortranarray(given.copy())
    missing = np.asfortranarray(missing.astype(np.int32))
    tools.reorder_missing_vector(actual, missing, inplace=True)
    assert_equal(actual, desired)
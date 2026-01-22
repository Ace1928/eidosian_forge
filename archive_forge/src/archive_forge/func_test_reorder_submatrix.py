import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
def test_reorder_submatrix():
    nobs = 5
    k_endog = 3
    missing = np.zeros((k_endog, nobs))
    missing[0, 0] = 1
    missing[:2, 1] = 1
    missing[0, 2] = 1
    missing[2, 2] = 1
    missing[1, 3] = 1
    missing[2, 4] = 1
    given = np.zeros((k_endog, k_endog, nobs))
    given[:, :, :] = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])[:, :, np.newaxis]
    desired = given.copy()
    given[:, :, 0] = np.array([[22, 23, 0], [32, 33, 0], [0, 0, 0]])
    desired[0, :, 0] = 0
    desired[:, 0, 0] = 0
    given[:, :, 1] = np.array([[33, 0, 0], [0, 0, 0], [0, 0, 0]])
    desired[:2, :, 1] = 0
    desired[:, :2, 1] = 0
    given[:, :, 2] = np.array([[22, 0, 0], [0, 0, 0], [0, 0, 0]])
    desired[0, :, 2] = 0
    desired[:, 0, 2] = 0
    desired[2, :, 2] = 0
    desired[:, 2, 2] = 0
    given[:, :, 3] = np.array([[11, 13, 0], [31, 33, 0], [0, 0, 0]])
    desired[1, :, 3] = 0
    desired[:, 1, 3] = 0
    given[:, :, 4] = np.array([[11, 12, 0], [21, 22, 0], [0, 0, 0]])
    desired[2, :, 4] = 0
    desired[:, 2, 4] = 0
    actual = np.asfortranarray(given)
    missing = np.asfortranarray(missing.astype(np.int32))
    tools.reorder_missing_matrix(actual, missing, True, True, False, inplace=True)
    assert_equal(actual, desired)
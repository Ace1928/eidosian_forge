import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
def test_copy_index_vector():
    nobs = 5
    k_endog = 3
    index = np.zeros((k_endog, nobs))
    index[0, 0] = 1
    index[:2, 1] = 1
    index[0, 2] = 1
    index[2, 2] = 1
    index[1, 3] = 1
    index[2, 4] = 1
    A = np.zeros((k_endog, nobs))
    for t in range(nobs):
        for i in range(k_endog):
            if index[i, t]:
                A[i, t] = 1.0
    B = np.zeros((k_endog, nobs), order='F')
    index = np.asfortranarray(index.astype(np.int32))
    tools.copy_index_vector(A, B, index, inplace=True)
    assert_equal(B, A)
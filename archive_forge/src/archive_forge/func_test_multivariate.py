import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
def test_multivariate(self):
    a = tools.companion_matrix([1, -0.4, 0.5])
    q = np.diag([10.0, 5.0])
    actual = tools.solve_discrete_lyapunov(a, q)
    desired = solve_discrete_lyapunov(a, q)
    assert_allclose(actual, desired)
    a = tools.companion_matrix([1, -0.4 + 0.1j, 0.5])
    q = np.diag([10.0, 5.0])
    actual = tools.solve_discrete_lyapunov(a, q, complex_step=False)
    desired = self.solve_dicrete_lyapunov_direct(a, q, complex_step=False)
    assert_allclose(actual, desired)
    a = tools.companion_matrix([1, -0.4 + 0.1j, 0.5])
    q = np.diag([10.0, 5.0])
    actual = tools.solve_discrete_lyapunov(a, q, complex_step=True)
    desired = self.solve_dicrete_lyapunov_direct(a, q, complex_step=True)
    assert_allclose(actual, desired)
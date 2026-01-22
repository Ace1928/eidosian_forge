import os
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.linalg import solve_sylvester
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.linalg import block_diag, solve, LinAlgError
from scipy.sparse._sputils import matrix
def check_continuous_case(self, a, q):
    x = solve_continuous_lyapunov(a, q)
    assert_array_almost_equal(np.dot(a, x) + np.dot(x, a.conj().transpose()), q)
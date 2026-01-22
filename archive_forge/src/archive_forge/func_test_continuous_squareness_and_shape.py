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
def test_continuous_squareness_and_shape(self):
    nsq = np.ones((3, 2))
    sq = np.eye(3)
    assert_raises(ValueError, solve_continuous_lyapunov, nsq, sq)
    assert_raises(ValueError, solve_continuous_lyapunov, sq, nsq)
    assert_raises(ValueError, solve_continuous_lyapunov, sq, np.eye(2))
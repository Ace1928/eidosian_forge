import numpy as np
from scipy.linalg._solve_toeplitz import levinson
from scipy.linalg import solve, toeplitz, solve_toeplitz
from numpy.testing import assert_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
def test_wikipedia_counterexample():
    random = np.random.RandomState(1234)
    c = [2, 2, 1]
    y = random.randn(3)
    assert_raises(np.linalg.LinAlgError, solve_toeplitz, c, b=y)
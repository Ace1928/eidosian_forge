import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
def very_random_gen(seed=0):
    np.random.seed(seed)
    m_eq, m_ub, n = (10, 20, 50)
    c = np.random.rand(n) - 0.5
    A_ub = np.random.rand(m_ub, n) - 0.5
    b_ub = np.random.rand(m_ub) - 0.5
    A_eq = np.random.rand(m_eq, n) - 0.5
    b_eq = np.random.rand(m_eq) - 0.5
    lb = -np.random.rand(n)
    ub = np.random.rand(n)
    lb[lb < -np.random.rand()] = -np.inf
    ub[ub > np.random.rand()] = np.inf
    bounds = np.vstack((lb, ub)).T
    return (c, A_ub, b_ub, A_eq, b_eq, bounds)
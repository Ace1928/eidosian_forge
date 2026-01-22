import unittest
import numpy as np
import pytest
import scipy
import scipy.sparse as sp
import scipy.stats
from numpy import linalg as LA
import cvxpy as cp
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.errormsg import SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms.partial_optimize import partial_optimize
def test_vec_to_upper_tri(self) -> None:
    x = Variable(shape=(3,))
    X = cp.vec_to_upper_tri(x)
    x.value = np.array([1, 2, 3])
    actual = X.value
    expect = np.array([[1, 2], [0, 3]])
    assert np.allclose(actual, expect)
    y = Variable(shape=(1,))
    y.value = np.array([4])
    Y = cp.vec_to_upper_tri(y, strict=True)
    actual = Y.value
    expect = np.array([[0, 4], [0, 0]])
    assert np.allclose(actual, expect)
    A_expect = np.array([[0, 11, 12, 13], [0, 0, 16, 17], [0, 0, 0, 21], [0, 0, 0, 0]])
    a = np.array([11, 12, 13, 16, 17, 21])
    A_actual = cp.vec_to_upper_tri(a, strict=True).value
    assert np.allclose(A_actual, A_expect)
    with pytest.raises(ValueError, match='must be a triangular number'):
        cp.vec_to_upper_tri(cp.Variable(shape=4))
    with pytest.raises(ValueError, match='must be a triangular number'):
        cp.vec_to_upper_tri(cp.Variable(shape=4), strict=True)
    with pytest.raises(ValueError, match='must be a vector'):
        cp.vec_to_upper_tri(cp.Variable(shape=(2, 2)))
    assert np.allclose(cp.vec_to_upper_tri(np.arange(6)).value, cp.vec_to_upper_tri(np.arange(6).reshape(1, 6)).value)
    assert np.allclose(cp.vec_to_upper_tri(1, strict=True).value, np.array([[0, 1], [0, 0]]))
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
def test_partial_transpose(self) -> None:
    """
        Test out the partial_transpose atom.
        rho_ABC = rho_A \\otimes rho_B \\otimes rho_C.
        Here \\otimes signifies Kronecker product.
        Each rho_i is normalized, i.e. Tr(rho_i) = 1.
        """
    np.random.seed(1)
    rho_A = np.random.random((8, 8)) + 1j * np.random.random((8, 8))
    rho_A /= np.trace(rho_A)
    rho_B = np.random.random((6, 6)) + 1j * np.random.random((6, 6))
    rho_B /= np.trace(rho_B)
    rho_C = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
    rho_C /= np.trace(rho_C)
    rho_TC = np.kron(np.kron(rho_A, rho_B), rho_C.T)
    rho_TB = np.kron(np.kron(rho_A, rho_B.T), rho_C)
    rho_TA = np.kron(np.kron(rho_A.T, rho_B), rho_C)
    temp = np.kron(np.kron(rho_A, rho_B), rho_C)
    rho_ABC = cp.Variable(shape=temp.shape, complex=True)
    rho_ABC.value = temp
    rho_TC_test = cp.partial_transpose(rho_ABC, [8, 6, 4], axis=2)
    rho_TB_test = cp.partial_transpose(rho_ABC, [8, 6, 4], axis=1)
    rho_TA_test = cp.partial_transpose(rho_ABC, [8, 6, 4], axis=0)
    assert np.allclose(rho_TC_test.value, rho_TC)
    assert np.allclose(rho_TB_test.value, rho_TB)
    assert np.allclose(rho_TA_test.value, rho_TA)
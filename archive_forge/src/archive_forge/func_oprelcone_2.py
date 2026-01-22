import unittest
import numpy as np
import pytest
import scipy as sp
import cvxpy as cp
from cvxpy import settings as s
from cvxpy.atoms.affine.trace import trace
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as INSTALLED_MI
from cvxpy.reductions.solvers.defines import MI_SOCP_SOLVERS as MI_SOCP
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
def oprelcone_2(self) -> STH.SolverTestHelper:
    """
        This test uses the same idea from the tests with commutative matrices,
        instead, here, we make the input matrices to Dop, non-commutative,
        the same condition as before i.e. T=Dop(A, B) for (A, B, T) in OpRelEntrConeQuad
        (for an objective that is an increasing function of the eigenvalues) holds,
        the difference here then, is in how we compute the reference values, which
        has been done by assuming correctness of the original CVXQUAD matlab implementation
        """
    n, m, k = (4, 3, 3)
    U1 = np.array([[-0.05878522, -0.78378355, -0.49418311, -0.37149791], [0.67696027, -0.25733435, 0.59263364, -0.35254672], [0.43478177, 0.53648704, -0.54593428, -0.47444939], [0.59096015, -0.17788771, -0.32638042, 0.71595942]])
    U2 = np.array([[-0.42499169, 0.6887562, 0.55846178, 0.18198188], [-0.55478633, -0.7091174, 0.3884544, 0.19613213], [-0.55591804, 0.14358541, -0.72444644, 0.38146522], [0.4500548, -0.04637494, 0.11135968, 0.88481584]])
    a_diag = cp.Variable(shape=(n,), pos=True)
    b_diag = cp.Variable(shape=(n,), pos=True)
    A = U1 @ cp.diag(a_diag) @ U1.T
    B = U2 @ cp.diag(b_diag) @ U2.T
    T = cp.Variable(shape=(n, n), symmetric=True)
    a_lower = np.array([0.40683013, 1.34514597, 1.60057343, 2.13373667])
    a_upper = np.array([1.36158501, 1.61289351, 1.85065805, 3.06140939])
    b_lower = np.array([0.06858235, 0.36798274, 0.95956627, 1.16286541])
    b_upper = np.array([0.70446555, 1.16635299, 1.46126732, 1.81367755])
    con1 = cp.constraints.OpRelEntrConeQuad(A, B, T, m, k)
    con2 = a_lower <= a_diag
    con3 = a_diag <= a_upper
    con4 = b_lower <= b_diag
    con5 = b_diag <= b_upper
    con_pairs = [(con1, None), (con2, None), (con3, None), (con4, None), (con5, None)]
    obj = cp.Minimize(trace(T))
    expect_obj = 1.85476
    expect_T = np.array([[0.49316819, 0.20845265, 0.60474713, -0.5820242], [0.20845265, 0.31084053, 0.2264112, -0.8442255], [0.60474713, 0.2264112, 0.4687153, -0.85667283], [-0.5820242, -0.8442255, -0.85667283, 0.58206723]])
    obj_pair = (obj, expect_obj)
    var_pairs = [(T, expect_T)]
    sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
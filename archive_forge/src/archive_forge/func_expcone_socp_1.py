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
def expcone_socp_1(self) -> STH.SolverTestHelper:
    """
        A random risk-parity portfolio optimization problem.
        """
    sigma = np.array([[1.83, 1.79, 3.22], [1.79, 2.18, 3.18], [3.22, 3.18, 8.69]])
    L = np.linalg.cholesky(sigma)
    c = 0.75
    t = cp.Variable(name='t')
    x = cp.Variable(shape=(3,), name='x')
    s = cp.Variable(shape=(3,), name='s')
    e = cp.Constant(np.ones(3))
    objective = cp.Minimize(t - c * e @ s)
    con1 = cp.norm(L.T @ x, p=2) <= t
    con2 = ExpCone(s, e, x).as_quad_approx(5, 5)
    obj_pair = (objective, 4.0751197)
    var_pairs = [(x, np.array([0.57608346, 0.54315695, 0.28037716])), (s, np.array([-0.5515, -0.61036, -1.27161]))]
    con_pairs = [(con1, 1.0), (con2, [None, None, None])]
    sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
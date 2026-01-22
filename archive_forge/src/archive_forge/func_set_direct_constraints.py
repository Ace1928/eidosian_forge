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
@staticmethod
def set_direct_constraints(y, K_dir):
    constraints = []
    i = K_dir[a2d.FREE]
    if K_dir[a2d.NONNEG]:
        dim = K_dir[a2d.NONNEG]
        constraints.append(y[i:i + dim] >= 0)
        i += dim
    for dim in K_dir[a2d.SOC]:
        constraints.append(SOC(y[i], y[i + 1:i + dim]))
        i += dim
    if K_dir[a2d.EXP]:
        dim = 3 * K_dir[a2d.EXP]
        expr = cp.reshape(y[i:i + dim], (dim // 3, 3), order='C')
        constraints.append(ExpCone(expr[:, 0], expr[:, 1], expr[:, 2]))
        i += dim
    if K_dir[a2d.POW3D]:
        alpha = np.array(K_dir[a2d.POW3D])
        expr = cp.reshape(y[i:], (alpha.size, 3), order='C')
        constraints.append(PowCone3D(expr[:, 0], expr[:, 1], expr[:, 2], alpha))
    return constraints
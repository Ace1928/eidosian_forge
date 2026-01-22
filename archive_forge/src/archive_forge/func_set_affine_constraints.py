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
def set_affine_constraints(G, h, y, K_aff):
    constraints = []
    i = 0
    if K_aff[a2d.ZERO]:
        dim = K_aff[a2d.ZERO]
        constraints.append(G[i:i + dim, :] @ y == h[i:i + dim])
        i += dim
    if K_aff[a2d.NONNEG]:
        dim = K_aff[a2d.NONNEG]
        constraints.append(G[i:i + dim, :] @ y <= h[i:i + dim])
        i += dim
    for dim in K_aff[a2d.SOC]:
        expr = h[i:i + dim] - G[i:i + dim, :] @ y
        constraints.append(SOC(expr[0], expr[1:]))
        i += dim
    if K_aff[a2d.EXP]:
        dim = 3 * K_aff[a2d.EXP]
        expr = cp.reshape(h[i:i + dim] - G[i:i + dim, :] @ y, (dim // 3, 3), order='C')
        constraints.append(ExpCone(expr[:, 0], expr[:, 1], expr[:, 2]))
        i += dim
    if K_aff[a2d.POW3D]:
        alpha = np.array(K_aff[a2d.POW3D])
        expr = cp.reshape(h[i:] - G[i:, :] @ y, (alpha.size, 3), order='C')
        constraints.append(PowCone3D(expr[:, 0], expr[:, 1], expr[:, 2], alpha))
    return constraints